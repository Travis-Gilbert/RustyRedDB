use crate::format::{
    ContentHash, DeploymentId, FormatError, LogId, LogIntent, LogPosition, LogRecord, OpDigest,
    RecordId, PHYSICAL_LOG_PROTOCOL_V2, PHYSICAL_LOG_PROTOCOL_V3, PHYSICAL_LOG_PROTOCOL_VERSION,
};
use crate::storage::LogFile;
use crate::{
    BridgeTransport, CommitAuthority, EntryOrigin, EntryProvenance, VerificationState, WriteSet,
    WriterId,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
use crate::storage::native::NativeLogFile;

/// The log file's name inside its directory.
///
/// The browser backend resolves the same name inside OPFS, so the constant is
/// cross-target even while nothing on wasm32 reads it without the browser
/// feature.
#[cfg_attr(
    all(target_arch = "wasm32", not(feature = "browser-opfs")),
    allow(dead_code, reason = "the OPFS backend resolves this name")
)]
pub(crate) const LOG_FILENAME: &str = "theorem.oplog";
const LOG_MAGIC: &[u8; 8] = b"THOPLG02";
const HEADER_BYTES: u64 = 56;
const FRAME_CHECKSUM_BYTES: u64 = 32;
const MAX_FRAME_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CrashInjection {
    BeforeWrite,
    AfterLength,
    AfterPayload,
    AfterChecksumBeforeSync,
    AfterSync,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PhysicalLogOptions {
    pub crash_injection: Option<CrashInjection>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppendReceipt {
    pub deployment_id: DeploymentId,
    pub log_id: LogId,
    pub position: LogPosition,
    pub record_id: RecordId,
    pub record_hash: ContentHash,
    pub replayed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PinnedPrefix {
    pub deployment_id: DeploymentId,
    pub log_id: LogId,
    pub through: LogPosition,
    pub content_hash: ContentHash,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifyReport {
    pub deployment_id: DeploymentId,
    pub records: usize,
    pub through: LogPosition,
    pub content_hash: ContentHash,
}

/// The durable ledger file, its decoded records, and the digest index over
/// them.
///
/// Storage is reached only through [`crate::storage::LogFile`], so the same
/// ledger runs over a filesystem or a browser's OPFS; nothing here names a
/// platform store.
#[derive(Debug)]
pub struct PhysicalLog {
    directory: PathBuf,
    file: Box<dyn LogFile>,
    deployment_id: DeploymentId,
    log_id: LogId,
    records: Vec<LogRecord>,
    digest_positions: BTreeMap<OpDigest, usize>,
    options: PhysicalLogOptions,
}

impl PhysicalLog {
    pub fn open(
        directory: impl AsRef<Path>,
        deployment_id: DeploymentId,
    ) -> Result<Self, PhysicalLogError> {
        Self::open_with_options(directory, deployment_id, PhysicalLogOptions::default())
    }

    pub fn open_with_options(
        directory: impl AsRef<Path>,
        deployment_id: DeploymentId,
        options: PhysicalLogOptions,
    ) -> Result<Self, PhysicalLogError> {
        let directory = directory.as_ref().to_path_buf();
        let file = open_log_file(&directory)?;
        Self::open_with_storage(directory, deployment_id, options, file)
    }

    /// Opens a log over a caller-supplied store.
    ///
    /// The native `open` builds its store from the directory name; a browser
    /// build builds one from OPFS handles, which are obtained where the
    /// platform requires it (a Worker, per SPEC-THEOREM-APP-HOST-2.0's
    /// `opfs-backend`). Everything after the store is supplied is the same
    /// code on both.
    #[cfg_attr(
        all(target_arch = "wasm32", not(feature = "browser-opfs")),
        allow(
            dead_code,
            reason = "the OPFS backend calls this through the browser-opfs feature"
        )
    )]
    pub(crate) fn open_with_storage(
        directory: PathBuf,
        deployment_id: DeploymentId,
        options: PhysicalLogOptions,
        mut file: Box<dyn LogFile>,
    ) -> Result<Self, PhysicalLogError> {
        let log_id = if file.len()? == 0 {
            let log_id = LogId::new();
            file.append(LOG_MAGIC)?;
            file.append(&log_id.0.to_bytes())?;
            file.append(deployment_hash(&deployment_id).as_bytes())?;
            file.sync()?;
            file.sync_directory()?;
            log_id
        } else {
            read_header(file.as_mut(), &deployment_id)?
        };

        let (records, valid_length, torn_tail) =
            scan_records(file.as_mut(), &deployment_id, log_id)?;
        if torn_tail {
            file.truncate(valid_length)?;
            file.sync()?;
        }

        let digest_positions = records
            .iter()
            .enumerate()
            .map(|(index, record)| (record.op_digest.clone(), index))
            .collect();
        Ok(Self {
            directory,
            file,
            deployment_id,
            log_id,
            records,
            digest_positions,
            options,
        })
    }

    pub fn log_id(&self) -> LogId {
        self.log_id
    }

    pub fn deployment_id(&self) -> &DeploymentId {
        &self.deployment_id
    }

    pub fn latest_position(&self) -> LogPosition {
        self.records
            .last()
            .map_or(LogPosition::ZERO, |record| record.position)
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn records(&self) -> &[LogRecord] {
        &self.records
    }

    pub fn records_after(&self, position: LogPosition) -> &[LogRecord] {
        let first = self
            .records
            .partition_point(|record| record.position <= position);
        &self.records[first..]
    }

    pub fn receipt_for_digest(&self, digest: &OpDigest) -> Option<AppendReceipt> {
        self.digest_positions
            .get(digest)
            .map(|index| receipt_for_record(&self.records[*index], true))
    }

    pub fn record_for_digest(&self, digest: &OpDigest) -> Option<&LogRecord> {
        self.digest_positions
            .get(digest)
            .map(|index| &self.records[*index])
    }

    pub fn append(
        &mut self,
        intent: LogIntent,
        write_set: WriteSet,
        op_digest: OpDigest,
    ) -> Result<AppendReceipt, PhysicalLogError> {
        self.append_with_provenance(intent, write_set, op_digest, internal_provenance())
    }

    pub(crate) fn append_with_provenance(
        &mut self,
        intent: LogIntent,
        write_set: WriteSet,
        op_digest: OpDigest,
        provenance: EntryProvenance,
    ) -> Result<AppendReceipt, PhysicalLogError> {
        intent.validate()?;
        validate_v3_provenance(&provenance)?;
        if let Some(existing) = self.record_for_digest(&op_digest) {
            let provenance_matches = existing.protocol_version == PHYSICAL_LOG_PROTOCOL_V2
                || existing.provenance == provenance;
            if existing.intent == intent && existing.write_set == write_set && provenance_matches {
                return Ok(receipt_for_record(existing, true));
            }
            return Err(PhysicalLogError::DuplicateDigestMismatch(op_digest));
        }

        let position = self
            .latest_position()
            .next()
            .ok_or(PhysicalLogError::PositionExhausted)?;
        let payload_hash = hash_json(&intent.payload)?;
        let mut record = LogRecord {
            protocol_version: PHYSICAL_LOG_PROTOCOL_VERSION,
            deployment_id: self.deployment_id.clone(),
            log_id: self.log_id,
            position,
            record_id: RecordId::new(),
            previous_record_hash: self.records.last().map(|record| record.record_hash.clone()),
            payload_hash,
            record_hash: ContentHash::from_bytes([]),
            op_digest: op_digest.clone(),
            provenance,
            intent,
            write_set,
        };
        record.record_hash = canonical_record_hash(&record)?;
        let payload = serde_json::to_vec(&record)?;
        if payload.len() as u64 > MAX_FRAME_BYTES {
            return Err(PhysicalLogError::FrameTooLarge(payload.len() as u64));
        }
        let checksum = blake3::hash(&payload);

        self.maybe_crash(CrashInjection::BeforeWrite);
        self.file.append(&(payload.len() as u32).to_le_bytes())?;
        self.maybe_crash(CrashInjection::AfterLength);
        self.file.append(&payload)?;
        self.maybe_crash(CrashInjection::AfterPayload);
        self.file.append(checksum.as_bytes())?;
        self.maybe_crash(CrashInjection::AfterChecksumBeforeSync);
        self.file.sync()?;
        self.maybe_crash(CrashInjection::AfterSync);

        let receipt = receipt_for_record(&record, false);
        self.digest_positions.insert(op_digest, self.records.len());
        self.records.push(record);
        Ok(receipt)
    }

    pub fn pinned_prefix(&self, through: LogPosition) -> Result<PinnedPrefix, PhysicalLogError> {
        if through > self.latest_position() {
            return Err(PhysicalLogError::PositionOutOfRange {
                requested: through,
                latest: self.latest_position(),
            });
        }
        let content_hash = if through == LogPosition::ZERO {
            hash_header(&self.deployment_id, self.log_id)
        } else {
            self.records[(through.0 - 1) as usize].record_hash.clone()
        };
        Ok(PinnedPrefix {
            deployment_id: self.deployment_id.clone(),
            log_id: self.log_id,
            through,
            content_hash,
        })
    }

    pub fn verify(&self) -> Result<VerifyReport, PhysicalLogError> {
        let mut previous = None;
        for (index, record) in self.records.iter().enumerate() {
            validate_record(
                record,
                &self.deployment_id,
                self.log_id,
                LogPosition(index as u64 + 1),
                previous,
            )?;
            previous = Some(&record.record_hash);
        }
        let prefix = self.pinned_prefix(self.latest_position())?;
        Ok(VerifyReport {
            deployment_id: self.deployment_id.clone(),
            records: self.records.len(),
            through: prefix.through,
            content_hash: prefix.content_hash,
        })
    }

    /// The name this log was opened under.
    ///
    /// The filesystem backend reads it as the directory holding the log file;
    /// a platform backend resolves the same name inside the platform's own
    /// storage root.
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    pub fn set_crash_injection(&mut self, crash_injection: Option<CrashInjection>) {
        self.options.crash_injection = crash_injection;
    }

    fn maybe_crash(&self, stage: CrashInjection) {
        if self.options.crash_injection == Some(stage) {
            std::process::abort();
        }
    }
}

#[derive(Debug)]
pub enum PhysicalLogError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Format(FormatError),
    Locked(String),
    /// A storage-backend failure that is not representable as
    /// `std::io::Error`, such as an OPFS handle fault in a browser. The
    /// filesystem backend reports through [`Self::Io`] and never produces
    /// this variant.
    Storage(String),
    InvalidHeader(String),
    Corrupt {
        offset: u64,
        detail: String,
    },
    FrameTooLarge(u64),
    PositionExhausted,
    PositionOutOfRange {
        requested: LogPosition,
        latest: LogPosition,
    },
    DuplicateDigestMismatch(OpDigest),
    InvalidProvenance(String),
    DeploymentMismatch {
        expected: DeploymentId,
    },
}

impl fmt::Display for PhysicalLogError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "physical log I/O failed: {error}"),
            Self::Json(error) => write!(formatter, "physical log JSON failed: {error}"),
            Self::Format(error) => write!(formatter, "physical log intent refused: {error}"),
            Self::Locked(path) => write!(formatter, "physical log already has a writer: {path}"),
            Self::Storage(detail) => write!(formatter, "physical log storage failed: {detail}"),
            Self::InvalidHeader(detail) => {
                write!(formatter, "physical log header invalid: {detail}")
            }
            Self::Corrupt { offset, detail } => {
                write!(formatter, "physical log corrupt at byte {offset}: {detail}")
            }
            Self::FrameTooLarge(bytes) => write!(formatter, "physical log frame is {bytes} bytes"),
            Self::PositionExhausted => formatter.write_str("physical log position space exhausted"),
            Self::PositionOutOfRange { requested, latest } => write!(
                formatter,
                "physical log position {requested} is beyond latest position {latest}"
            ),
            Self::DuplicateDigestMismatch(digest) => write!(
                formatter,
                "operation digest {digest} was reused with different intent, write set, or provenance"
            ),
            Self::InvalidProvenance(detail) => {
                write!(formatter, "physical log provenance invalid: {detail}")
            }
            Self::DeploymentMismatch { expected } => write!(
                formatter,
                "physical log does not belong to deployment {expected}"
            ),
        }
    }
}

impl Error for PhysicalLogError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Format(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for PhysicalLogError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for PhysicalLogError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<FormatError> for PhysicalLogError {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}

/// The log file this build links against.
///
/// A wasm32 build has no filesystem behind `std::fs`, and its storage backend
/// cannot be reached from a synchronous `open`: OPFS sync access handles are
/// acquired in a worker, asynchronously. So this refuses by name, and says
/// which of the two cases the build is in.
#[cfg(not(target_arch = "wasm32"))]
fn open_log_file(directory: &Path) -> Result<Box<dyn LogFile>, PhysicalLogError> {
    Ok(Box::new(NativeLogFile::create_or_open(directory)?))
}

#[cfg(target_arch = "wasm32")]
fn open_log_file(_directory: &Path) -> Result<Box<dyn LogFile>, PhysicalLogError> {
    #[cfg(feature = "browser-opfs")]
    let detail = "a browser log opens asynchronously: OPFS sync access handles are acquired \
                  in a worker, so a browser build owns the log through the browser backend's \
                  open_in_worker rather than through this synchronous open";
    #[cfg(not(feature = "browser-opfs"))]
    let detail = "no browser log backend is installed in this build; rebuild with the \
                  browser-opfs feature for the OPFS backend of SPEC-THEOREM-APP-HOST-2.0";
    Err(PhysicalLogError::Storage(detail.to_string()))
}

fn read_header(
    file: &mut dyn LogFile,
    expected_deployment_id: &DeploymentId,
) -> Result<LogId, PhysicalLogError> {
    if file.len()? < HEADER_BYTES {
        return Err(PhysicalLogError::InvalidHeader(
            "file is shorter than the fixed header".to_string(),
        ));
    }
    let mut magic = [0_u8; 8];
    file.read_exact_at(0, &mut magic)?;
    if &magic != LOG_MAGIC {
        return Err(PhysicalLogError::InvalidHeader(format!(
            "unexpected magic {magic:?}"
        )));
    }
    let mut log_id = [0_u8; 16];
    file.read_exact_at(8, &mut log_id)?;
    let mut stored_deployment_hash = [0_u8; 32];
    file.read_exact_at(24, &mut stored_deployment_hash)?;
    if stored_deployment_hash != *deployment_hash(expected_deployment_id).as_bytes() {
        return Err(PhysicalLogError::DeploymentMismatch {
            expected: expected_deployment_id.clone(),
        });
    }
    Ok(LogId(ulid::Ulid::from_bytes(log_id)))
}

fn scan_records(
    file: &mut dyn LogFile,
    deployment_id: &DeploymentId,
    log_id: LogId,
) -> Result<(Vec<LogRecord>, u64, bool), PhysicalLogError> {
    let file_length = file.len()?;
    let mut offset = HEADER_BYTES;
    let mut records = Vec::new();
    let mut previous = None;

    while offset < file_length {
        if file_length - offset < 4 {
            return Ok((records, offset, true));
        }
        let frame_offset = offset;
        let mut length_bytes = [0_u8; 4];
        file.read_exact_at(frame_offset, &mut length_bytes)?;
        let payload_length = u32::from_le_bytes(length_bytes) as u64;
        if payload_length > MAX_FRAME_BYTES {
            return Err(PhysicalLogError::Corrupt {
                offset: frame_offset,
                detail: format!("frame length {payload_length} exceeds limit"),
            });
        }
        let frame_length = 4 + payload_length + FRAME_CHECKSUM_BYTES;
        if file_length - frame_offset < frame_length {
            return Ok((records, frame_offset, true));
        }

        let mut payload = vec![0_u8; payload_length as usize];
        file.read_exact_at(frame_offset + 4, &mut payload)?;
        let mut checksum = [0_u8; 32];
        file.read_exact_at(frame_offset + 4 + payload_length, &mut checksum)?;
        if blake3::hash(&payload).as_bytes() != &checksum {
            return Err(PhysicalLogError::Corrupt {
                offset: frame_offset,
                detail: "frame checksum mismatch".to_string(),
            });
        }
        let record: LogRecord =
            serde_json::from_slice(&payload).map_err(|error| PhysicalLogError::Corrupt {
                offset: frame_offset,
                detail: format!("record decode failed: {error}"),
            })?;
        validate_record(
            &record,
            deployment_id,
            log_id,
            LogPosition(records.len() as u64 + 1),
            previous,
        )
        .map_err(|error| PhysicalLogError::Corrupt {
            offset: frame_offset,
            detail: error.to_string(),
        })?;
        offset = frame_offset + frame_length;
        records.push(record);
        previous = records.last().map(|record| &record.record_hash);
    }
    Ok((records, offset, false))
}

fn validate_record(
    record: &LogRecord,
    expected_deployment_id: &DeploymentId,
    expected_log_id: LogId,
    expected_position: LogPosition,
    expected_previous_hash: Option<&ContentHash>,
) -> Result<(), PhysicalLogError> {
    if !matches!(
        record.protocol_version,
        PHYSICAL_LOG_PROTOCOL_V2 | PHYSICAL_LOG_PROTOCOL_V3
    ) {
        return Err(PhysicalLogError::InvalidHeader(format!(
            "unsupported record protocol {}",
            record.protocol_version
        )));
    }
    match record.protocol_version {
        PHYSICAL_LOG_PROTOCOL_V2 if !record.provenance.is_explicit_legacy() => {
            return Err(PhysicalLogError::InvalidHeader(format!(
                "v2 record {} carries provenance outside its canonical hash",
                record.position
            )));
        }
        PHYSICAL_LOG_PROTOCOL_V3 => validate_v3_provenance(&record.provenance)?,
        PHYSICAL_LOG_PROTOCOL_V2 => {}
        _ => unreachable!("supported protocols were checked above"),
    }
    if &record.deployment_id != expected_deployment_id {
        return Err(PhysicalLogError::DeploymentMismatch {
            expected: expected_deployment_id.clone(),
        });
    }
    if record.log_id != expected_log_id {
        return Err(PhysicalLogError::InvalidHeader(format!(
            "record log {} does not match header {}",
            record.log_id, expected_log_id
        )));
    }
    if record.position != expected_position {
        return Err(PhysicalLogError::InvalidHeader(format!(
            "record position {} does not match expected {}",
            record.position, expected_position
        )));
    }
    if record.previous_record_hash.as_ref() != expected_previous_hash {
        return Err(PhysicalLogError::InvalidHeader(format!(
            "record {} previous hash does not match its predecessor",
            record.position
        )));
    }
    record.intent.validate()?;
    if record.payload_hash != hash_json(&record.intent.payload)? {
        return Err(PhysicalLogError::InvalidHeader(format!(
            "record {} payload hash mismatch",
            record.position
        )));
    }
    if record.record_hash != canonical_record_hash(record)? {
        return Err(PhysicalLogError::InvalidHeader(format!(
            "record {} chain hash mismatch",
            record.position
        )));
    }
    Ok(())
}

#[derive(Serialize)]
struct RecordHashInputV2<'a> {
    protocol_version: u32,
    deployment_id: &'a DeploymentId,
    log_id: LogId,
    position: LogPosition,
    record_id: RecordId,
    previous_record_hash: &'a Option<ContentHash>,
    payload_hash: &'a ContentHash,
    op_digest: &'a OpDigest,
    intent: &'a LogIntent,
    write_set: &'a WriteSet,
}

#[derive(Serialize)]
struct RecordHashInputV3<'a> {
    protocol_version: u32,
    deployment_id: &'a DeploymentId,
    log_id: LogId,
    position: LogPosition,
    record_id: RecordId,
    previous_record_hash: &'a Option<ContentHash>,
    payload_hash: &'a ContentHash,
    op_digest: &'a OpDigest,
    provenance: &'a EntryProvenance,
    intent: &'a LogIntent,
    write_set: &'a WriteSet,
}

/// Recompute the canonical chain hash selected by a record's protocol.
///
/// This is public so migration and verification tooling can audit stored
/// records without reimplementing either historical preimage.
pub fn canonical_record_hash(record: &LogRecord) -> Result<ContentHash, PhysicalLogError> {
    match record.protocol_version {
        PHYSICAL_LOG_PROTOCOL_V2 => hash_json(&RecordHashInputV2 {
            protocol_version: record.protocol_version,
            deployment_id: &record.deployment_id,
            log_id: record.log_id,
            position: record.position,
            record_id: record.record_id,
            previous_record_hash: &record.previous_record_hash,
            payload_hash: &record.payload_hash,
            op_digest: &record.op_digest,
            intent: &record.intent,
            write_set: &record.write_set,
        }),
        PHYSICAL_LOG_PROTOCOL_V3 => hash_json(&RecordHashInputV3 {
            protocol_version: record.protocol_version,
            deployment_id: &record.deployment_id,
            log_id: record.log_id,
            position: record.position,
            record_id: record.record_id,
            previous_record_hash: &record.previous_record_hash,
            payload_hash: &record.payload_hash,
            op_digest: &record.op_digest,
            provenance: &record.provenance,
            intent: &record.intent,
            write_set: &record.write_set,
        }),
        protocol => Err(PhysicalLogError::InvalidHeader(format!(
            "unsupported record protocol {protocol}"
        ))),
    }
}

pub(crate) fn validate_v3_provenance(provenance: &EntryProvenance) -> Result<(), PhysicalLogError> {
    if provenance.is_explicit_legacy() {
        return Err(PhysicalLogError::InvalidProvenance(
            "v3 append requires an admitted non-legacy origin and authority".to_string(),
        ));
    }
    if !provenance.is_authority_consistent() {
        return Err(PhysicalLogError::InvalidProvenance(format!(
            "authority {:?} cannot commit origin {:?}",
            provenance.authority, provenance.origin
        )));
    }
    if provenance.transport.is_none() {
        return Err(PhysicalLogError::InvalidProvenance(
            "v3 append requires a transport".to_string(),
        ));
    }
    if provenance.writer.is_none() {
        return Err(PhysicalLogError::InvalidProvenance(
            "v3 append requires a writer identity".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn internal_provenance() -> EntryProvenance {
    EntryProvenance {
        origin: EntryOrigin::Internal,
        authority: CommitAuthority::Internal,
        transport: Some(BridgeTransport::InProcess),
        verification: VerificationState::Unverified,
        writer: Some(
            WriterId::new("theorem-oplog:physical")
                .expect("the static physical-log writer identity is valid"),
        ),
    }
}

fn hash_json(value: &impl Serialize) -> Result<ContentHash, PhysicalLogError> {
    Ok(ContentHash::from_bytes(serde_json::to_vec(value)?))
}

fn hash_header(deployment_id: &DeploymentId, log_id: LogId) -> ContentHash {
    let mut bytes = Vec::with_capacity(HEADER_BYTES as usize);
    bytes.extend_from_slice(LOG_MAGIC);
    bytes.extend_from_slice(&log_id.0.to_bytes());
    bytes.extend_from_slice(deployment_hash(deployment_id).as_bytes());
    ContentHash::from_bytes(bytes)
}

fn deployment_hash(deployment_id: &DeploymentId) -> blake3::Hash {
    blake3::hash(deployment_id.as_str().as_bytes())
}

fn receipt_for_record(record: &LogRecord, replayed: bool) -> AppendReceipt {
    AppendReceipt {
        deployment_id: record.deployment_id.clone(),
        log_id: record.log_id,
        position: record.position,
        record_id: record.record_id,
        record_hash: record.record_hash.clone(),
        replayed,
    }
}
