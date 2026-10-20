//! The `.RR` file header: the block that makes a RustyRed database
//! self-identifying.
//!
//! The journal frame stream already proves its own integrity. Every frame
//! carries `RRCJ` magic, a CRC32 and a SHA-256 content address, and recovery
//! already truncates a torn tail without losing a committed root. What the
//! stream cannot do is answer "what is this file?" without parsing a record,
//! because `RRCJ` is per-record: offset zero is the first frame, not an
//! identity block.
//!
//! SPEC-THEOREM-DESKTOP-SHELL-SERVO-RR-1.0 S29 requires exactly that missing
//! block: magic, a format version, a capability version, a database identity
//! and integrity metadata, with "Extension alone is never trusted".
//!
//! So an `.RR` file is this fixed [`RR_HEADER_LEN`] block at offset zero
//! followed by the ordinary journal frame stream. Durability, torn-tail
//! recovery and atomic root publication stay in the code that already survives
//! its own crash tests. Only identification is new.
//!
//! # The two versions mean different things
//!
//! [`RR_FORMAT_VERSION`] names the physical layout of this block. A reader that
//! does not know the layout refuses the file outright, because it cannot even
//! locate the fields it would need to decide anything else.
//!
//! [`RR_CAPABILITY_VERSION`] names the features the body is allowed to use. A
//! reader that knows the layout but not the capabilities still refuses a
//! writable open, because appending beside a feature it cannot interpret risks
//! corrupting it, and still permits [`super::RustyRed::open_read_only`]. Two
//! fields exist because these two failures deserve different answers.
//!
//! # Validation order, and the rule it depends on
//!
//! Validation runs magic, then digest, then versions. The digest comes before
//! the versions deliberately. It is the only thing that separates a damaged
//! block from a well-formed one, and reading a version field out of a torn
//! block reports the wrong failure: "unsupported format 51966" instead of
//! "corrupt". Tests that tear the header want the second answer.
//!
//! That ordering is sound only while the digest's location is fixed by the
//! magic alone. So a future layout that moves the digest MUST take a new magic
//! rather than only bumping [`RR_FORMAT_VERSION`].

use rand_chacha::rand_core::{OsRng, RngCore};
use sha2::{Digest, Sha256};
use theorem_surface_contracts::desktop::DatabaseId;

use crate::graph::error::{GraphStoreError, GraphStoreResult};

/// Identifies an `.RR` file by content rather than by name.
///
/// S29's "Extension alone is never trusted" cuts both ways: a file named
/// `Travis.RR` whose first eight bytes are not this is refused, and a file with
/// any other name whose first eight bytes are this is accepted.
pub const RR_MAGIC: [u8; 8] = *b"THEORRDB";

/// Fixed size of the header block. The frame stream starts at this offset.
///
/// One 4 KiB block, so the header occupies whole pages on every filesystem this
/// runs on and a header write cannot share a sector with body data.
pub const RR_HEADER_LEN: u64 = 4096;

/// Physical layout of the header block. See the module docs before bumping it.
pub const RR_FORMAT_VERSION: u16 = 1;

/// Features the body is allowed to use.
pub const RR_CAPABILITY_VERSION: u16 = 1;

/// The body is a `RRCJ` journal frame stream, recoverable by the journal
/// scanner. This is the recovery half of S29's "integrity/recovery metadata":
/// the digest says the header is intact, this says how to rebuild the body.
pub const RR_BODY_FORMAT_JOURNAL: u16 = 1;

/// Longest `DatabaseId` the fixed block can hold.
pub const RR_MAX_DATABASE_ID_LEN: usize = 256;

/// Domain separation byte for the header digest.
///
/// Seven is the one value the crate's existing domain bytes leave free: 1-6 are
/// taken by `versioned_graph` and `chunk_store`, 8-9 by `doc_tree`, 10-12 and
/// 14-15 by `object_store`, and 13 by `file_payload`.
const DOMAIN_RR_HEADER: u8 = 7;

const MAGIC_AT: usize = 0;
const FORMAT_VERSION_AT: usize = 8;
const CAPABILITY_VERSION_AT: usize = 10;
const BODY_FORMAT_AT: usize = 12;
const RESERVED_AT: usize = 14;
const DATABASE_ID_LEN_AT: usize = 16;
const DATABASE_ID_AT: usize = 20;
const DIGEST_LEN: usize = 32;
const DIGEST_AT: usize = RR_HEADER_LEN as usize - DIGEST_LEN;

/// Mint a fresh database identity.
///
/// 128 bits from the operating system's CSPRNG. It has to be real entropy
/// rather than a timestamp or a path hash, because `DatabaseId` is documented
/// as stable across copy, move and rename, so identities travel between
/// machines and a derived id could collide there.
#[must_use]
pub fn mint_database_id() -> DatabaseId {
    let mut bytes = [0_u8; 16];
    OsRng.fill_bytes(&mut bytes);
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        hex.push_str(&format!("{byte:02x}"));
    }
    DatabaseId::new(hex)
}

/// The decoded header block of an `.RR` file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RrHeader {
    format_version: u16,
    capability_version: u16,
    body_format: u16,
    database: DatabaseId,
}

impl RrHeader {
    /// A header for a database created by this build.
    pub fn new(database: DatabaseId) -> GraphStoreResult<Self> {
        if database.as_str().is_empty() {
            return Err(GraphStoreError::new(
                "rr_database_id_invalid",
                "database identity is empty",
            ));
        }
        if database.as_str().len() > RR_MAX_DATABASE_ID_LEN {
            return Err(GraphStoreError::new(
                "rr_database_id_invalid",
                format!(
                    "database identity is {} bytes, over the {RR_MAX_DATABASE_ID_LEN} byte header limit",
                    database.as_str().len()
                ),
            ));
        }
        Ok(Self {
            format_version: RR_FORMAT_VERSION,
            capability_version: RR_CAPABILITY_VERSION,
            body_format: RR_BODY_FORMAT_JOURNAL,
            database,
        })
    }

    /// The identity this file carries. Survives copy, move and rename because
    /// it lives in the file rather than being derived from its path.
    #[must_use]
    pub const fn database(&self) -> &DatabaseId {
        &self.database
    }

    #[must_use]
    pub const fn format_version(&self) -> u16 {
        self.format_version
    }

    #[must_use]
    pub const fn capability_version(&self) -> u16 {
        self.capability_version
    }

    #[must_use]
    pub const fn body_format(&self) -> u16 {
        self.body_format
    }

    /// Serialize to the fixed block written at offset zero.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut block = vec![0_u8; RR_HEADER_LEN as usize];
        block[MAGIC_AT..MAGIC_AT + RR_MAGIC.len()].copy_from_slice(&RR_MAGIC);
        block[FORMAT_VERSION_AT..FORMAT_VERSION_AT + 2]
            .copy_from_slice(&self.format_version.to_le_bytes());
        block[CAPABILITY_VERSION_AT..CAPABILITY_VERSION_AT + 2]
            .copy_from_slice(&self.capability_version.to_le_bytes());
        block[BODY_FORMAT_AT..BODY_FORMAT_AT + 2].copy_from_slice(&self.body_format.to_le_bytes());
        block[RESERVED_AT..RESERVED_AT + 2].copy_from_slice(&0_u16.to_le_bytes());
        let id = self.database.as_str().as_bytes();
        let id_len = u32::try_from(id.len()).unwrap_or(u32::MAX);
        block[DATABASE_ID_LEN_AT..DATABASE_ID_LEN_AT + 4].copy_from_slice(&id_len.to_le_bytes());
        block[DATABASE_ID_AT..DATABASE_ID_AT + id.len()].copy_from_slice(id);
        let digest = header_digest(&block[..DIGEST_AT]);
        block[DIGEST_AT..].copy_from_slice(&digest);
        block
    }

    /// Read a header out of the first [`RR_HEADER_LEN`] bytes of a file.
    ///
    /// Refuses by name at each stage so a caller learns which of "not a
    /// database", "damaged", and "from a newer build" it hit.
    pub fn decode(block: &[u8]) -> GraphStoreResult<Self> {
        if block.len() < RR_HEADER_LEN as usize {
            return Err(GraphStoreError::new(
                "rr_header_truncated",
                format!(
                    "file holds {} bytes, short of the {RR_HEADER_LEN} byte header block",
                    block.len()
                ),
            ));
        }
        if block[MAGIC_AT..MAGIC_AT + RR_MAGIC.len()] != RR_MAGIC {
            return Err(GraphStoreError::new(
                "rr_not_a_database",
                "file does not start with the RR magic, whatever its extension says",
            ));
        }
        let expected = header_digest(&block[..DIGEST_AT]);
        if block[DIGEST_AT..DIGEST_AT + DIGEST_LEN] != expected {
            return Err(GraphStoreError::new(
                "rr_header_corrupt",
                "RR header digest does not match its contents",
            ));
        }

        let format_version = read_u16(block, FORMAT_VERSION_AT);
        if format_version != RR_FORMAT_VERSION {
            return Err(GraphStoreError::new(
                "rr_format_unsupported",
                format!(
                    "RR header declares format {format_version}, this build reads {RR_FORMAT_VERSION}"
                ),
            ));
        }
        let capability_version = read_u16(block, CAPABILITY_VERSION_AT);
        let body_format = read_u16(block, BODY_FORMAT_AT);
        if body_format != RR_BODY_FORMAT_JOURNAL {
            return Err(GraphStoreError::new(
                "rr_body_format_unsupported",
                format!(
                    "RR header declares body format {body_format}, this build recovers {RR_BODY_FORMAT_JOURNAL}"
                ),
            ));
        }

        let id_len = usize::try_from(read_u32(block, DATABASE_ID_LEN_AT)).unwrap_or(usize::MAX);
        if id_len == 0 || id_len > RR_MAX_DATABASE_ID_LEN {
            return Err(GraphStoreError::new(
                "rr_database_id_invalid",
                format!("RR header declares a {id_len} byte database identity"),
            ));
        }
        let id_bytes = &block[DATABASE_ID_AT..DATABASE_ID_AT + id_len];
        let database = std::str::from_utf8(id_bytes).map_err(|error| {
            GraphStoreError::new(
                "rr_database_id_invalid",
                format!("RR database identity is not UTF-8: {error}"),
            )
        })?;

        Ok(Self {
            format_version,
            capability_version,
            body_format,
            database: DatabaseId::new(database),
        })
    }

    /// Whether this build may open the file writable.
    ///
    /// A body written with capabilities this build does not implement is
    /// readable but not safely appendable, so the refusal is scoped to writes
    /// rather than to the whole file.
    #[must_use]
    pub const fn writable_by_this_build(&self) -> bool {
        self.capability_version <= RR_CAPABILITY_VERSION
    }
}

fn header_digest(bytes: &[u8]) -> [u8; DIGEST_LEN] {
    let mut hasher = Sha256::new();
    hasher.update([DOMAIN_RR_HEADER]);
    hasher.update(bytes);
    hasher.finalize().into()
}

fn read_u16(block: &[u8], at: usize) -> u16 {
    let mut bytes = [0_u8; 2];
    bytes.copy_from_slice(&block[at..at + 2]);
    u16::from_le_bytes(bytes)
}

fn read_u32(block: &[u8], at: usize) -> u32 {
    let mut bytes = [0_u8; 4];
    bytes.copy_from_slice(&block[at..at + 4]);
    u32::from_le_bytes(bytes)
}
