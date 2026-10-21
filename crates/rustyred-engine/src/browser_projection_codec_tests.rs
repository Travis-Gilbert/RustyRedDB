use super::*;

fn directory() -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix("projection-v2-")
        .tempdir_in(std::env::temp_dir())
        .unwrap()
}

fn file(path: &Path) -> Box<dyn LogFile> {
    Box::new(
        theorem_oplog::NativeLogFile::create_or_open_named(path, BROWSER_CHUNK_JOURNAL_FILE)
            .unwrap(),
    )
}

fn publish(store: &ProjectionChunkStore, bytes: &[u8]) -> GraphStoreResult<Hash> {
    let address = Hash::of(bytes);
    store.write(&[(address, bytes)])?;
    store.commit_catalog_updates(
        &BTreeMap::from([("graph".into(), Some(address))]),
        CommitMeta {
            parents: Vec::new(),
            author: "projection-codec-test".into(),
            message: "publish canonical source".into(),
            timestamp_unix_ms: 1,
            graph_version: 1,
        },
    )?;
    Ok(address)
}

#[test]
fn compressed_projection_reopens_canonical_bytes_and_admits_decoded_length() -> GraphStoreResult<()>
{
    let dir = directory();
    let bytes = "same canonical source with repeated tree keys; "
        .repeat(2000)
        .into_bytes();
    let store = ProjectionChunkStore::open_with_compressed_chunks(file(dir.path()), None)?;
    let address = publish(&store, &bytes)?;
    let root = store.read_root()?;
    assert_eq!(store.indexed_chunk_len(address)?, Some(bytes.len() as u64));
    assert_eq!(
        store
            .get_bounded(address, bytes.len() - 1)
            .unwrap_err()
            .code,
        "chunk_read_bound_refused"
    );
    assert_eq!(
        store.get_bounded(address, bytes.len())?.unwrap().as_ref(),
        bytes
    );
    assert!(store.storage_snapshot()?.log_bytes < bytes.len() as u64 / 4);
    drop(store);
    let reopened = ProjectionChunkStore::open(file(dir.path()))?;
    assert_eq!(reopened.read_root()?, root);
    assert_eq!(reopened.get(address)?.unwrap().as_ref(), bytes);
    publish(&reopened, &b"a later canonical source".repeat(500))?;
    drop(reopened);
    assert_eq!(
        std::fs::read(dir.path().join(BROWSER_CHUNK_JOURNAL_FILE)).unwrap()[8],
        COMPRESSED_JOURNAL_VERSION
    );
    Ok(())
}

#[test]
fn compressed_opener_preserves_existing_v1_bytes_and_writer_version() -> GraphStoreResult<()> {
    let dir = directory();
    let bytes = b"retained legacy source".repeat(500);
    let legacy = ProjectionChunkStore::open(file(dir.path()))?;
    let address = publish(&legacy, &bytes)?;
    drop(legacy);
    let path = dir.path().join(BROWSER_CHUNK_JOURNAL_FILE);
    let before = std::fs::read(&path).unwrap();
    let reopened = ProjectionChunkStore::open_with_compressed_chunks(file(dir.path()), None)?;
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(reopened.get(address)?.unwrap().as_ref(), bytes);
    publish(&reopened, &b"later legacy source".repeat(500))?;
    drop(reopened);
    let mut disk = file(dir.path());
    let data = std::fs::read(&path).unwrap();
    assert_eq!(data[8], JOURNAL_VERSION);
    let mut offset = JOURNAL_HEADER_BYTES;
    while offset < data.len() as u64 {
        let FrameRead::Valid(frame) = read_frame(disk.as_mut(), offset, data.len() as u64)? else {
            panic!("valid V1 journal");
        };
        assert_ne!(frame.kind, FrameKind::CompressedChunk);
        offset += FRAME_HEADER_BYTES + frame.payload.len() as u64;
    }
    Ok(())
}

#[test]
fn unknown_projection_version_refuses_before_modifying_source() -> GraphStoreResult<()> {
    let dir = directory();
    let store = ProjectionChunkStore::open_with_compressed_chunks(file(dir.path()), None)?;
    publish(&store, &b"retained source".repeat(200))?;
    drop(store);
    let path = dir.path().join(BROWSER_CHUNK_JOURNAL_FILE);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[8] = COMPRESSED_JOURNAL_VERSION + 1;
    std::fs::write(&path, &bytes).unwrap();
    for compressed in [false, true] {
        let result = if compressed {
            ProjectionChunkStore::open_with_compressed_chunks(file(dir.path()), None)
        } else {
            ProjectionChunkStore::open(file(dir.path()))
        };
        assert_eq!(
            result.unwrap_err().code,
            "chunk_store_browser_unsupported_format"
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }
    Ok(())
}

#[test]
fn v1_refuses_v2_frames_before_any_suffix_repair() -> GraphStoreResult<()> {
    let dir = directory();
    let store = ProjectionChunkStore::open(file(dir.path()))?;
    publish(&store, b"acknowledged V1 source")?;
    drop(store);
    let path = dir.path().join(BROWSER_CHUNK_JOURNAL_FILE);
    let mut bytes = std::fs::read(&path).unwrap();
    // Even an incomplete V2-only header is unsupported, never a torn V1 tail.
    bytes.push(FrameKind::CompressedChunk as u8);
    std::fs::write(&path, &bytes).unwrap();
    let error =
        ProjectionChunkStore::open_with_compressed_chunks(file(dir.path()), None).unwrap_err();
    assert_eq!(error.code, "chunk_store_browser_unsupported_format");
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    Ok(())
}

#[test]
fn compressed_envelope_rejects_oversize_forged_hash_and_wrong_length() -> GraphStoreResult<()> {
    let bytes = b"canonical repeating payload".repeat(500);
    let envelope = codec::encode(&bytes)?.unwrap();
    assert_eq!(codec::decode(&envelope)?, bytes);
    let mut bad = envelope.clone();
    bad[..4].copy_from_slice(&(MAX_FRAME_PAYLOAD + 1).to_le_bytes());
    assert_eq!(
        codec::decode(&bad).unwrap_err().code,
        "chunk_store_browser_payload_too_large"
    );
    let mut bad = envelope.clone();
    bad[4] ^= 1;
    assert!(codec::decode(&bad).is_err());
    let mut bad = envelope.clone();
    bad[..4].copy_from_slice(&((bytes.len() - 1) as u32).to_le_bytes());
    assert!(codec::decode(&bad).is_err());
    let mut bad = envelope.clone();
    bad[..4].copy_from_slice(&((bytes.len() + 1) as u32).to_le_bytes());
    assert!(codec::decode(&bad).is_err());
    let mut bad = envelope;
    bad.truncate(36);
    assert!(codec::decode(&bad).is_err());
    Ok(())
}

#[test]
fn bounded_read_rechecks_disk_decoded_length_before_decoder_allocation() -> GraphStoreResult<()> {
    use std::io::{Seek, SeekFrom, Write};
    let dir = directory();
    let bytes = b"admitted canonical source".repeat(500);
    let store = ProjectionChunkStore::open_with_compressed_chunks(file(dir.path()), None)?;
    let address = publish(&store, &bytes)?;
    let location = { *store.lock_state()?.index.get(&address).unwrap() };
    assert!(location.compressed);
    let mut disk = std::fs::OpenOptions::new()
        .write(true)
        .open(dir.path().join(BROWSER_CHUNK_JOURNAL_FILE))
        .unwrap();
    disk.seek(SeekFrom::Start(location.payload_offset)).unwrap();
    disk.write_all(&MAX_FRAME_PAYLOAD.to_le_bytes()).unwrap();
    disk.sync_all().unwrap();
    codec::DECODE_CAPACITY.with(|capacity| capacity.set(None));
    let refusal = store.get_bounded(address, bytes.len()).unwrap_err();
    assert_eq!(
        codec::DECODE_CAPACITY.with(std::cell::Cell::get),
        None,
        "changed length must refuse before calling the allocating decoder"
    );
    assert_eq!(refusal.code, "chunk_store_browser_chunk_length_mismatch");
    Ok(())
}

#[test]
fn corrupt_compressed_chunk_before_commit_refuses_without_truncation() -> GraphStoreResult<()> {
    let dir = directory();
    let store = ProjectionChunkStore::open_with_compressed_chunks(file(dir.path()), None)?;
    publish(
        &store,
        &b"canonical source must survive corruption diagnosis".repeat(500),
    )?;
    drop(store);
    let path = dir.path().join(BROWSER_CHUNK_JOURNAL_FILE);
    let mut data = std::fs::read(&path).unwrap();
    assert_eq!(data[9], FrameKind::CompressedChunk as u8);
    let len = u32::from_le_bytes(data[10..14].try_into().unwrap()) as usize;
    // Keep the frame CRC valid: canonical decoded hash must catch this too.
    data[18 + 4] ^= 1;
    let crc = crc32fast::hash(&data[18..18 + len]);
    data[14..18].copy_from_slice(&crc.to_le_bytes());
    std::fs::write(&path, &data).unwrap();
    let error = ProjectionChunkStore::open(file(dir.path())).unwrap_err();
    assert_eq!(error.code, "chunk_store_browser_corrupt_committed_data");
    assert_eq!(std::fs::read(&path).unwrap(), data);
    Ok(())
}

#[test]
fn compressed_publication_flush_failure_preserves_previous_acknowledged_root(
) -> GraphStoreResult<()> {
    let dir = directory();
    let store = ProjectionChunkStore::open_with_compressed_chunks(file(dir.path()), None)?;
    let bytes = b"acknowledged source".repeat(500);
    let acknowledged = publish(&store, &bytes)?;
    let before = store.read_root()?;
    store.fail_next_publication_flush_for_test()?;
    assert!(publish(&store, &b"unacknowledged source".repeat(500)).is_err());
    assert_eq!(store.read_root()?, before);
    drop(store);
    let reopened = ProjectionChunkStore::open(file(dir.path()))?;
    assert_eq!(reopened.read_root()?, before);
    assert_eq!(reopened.get(acknowledged)?.unwrap().as_ref(), bytes);
    Ok(())
}

proptest::proptest! {
    #[test]
    fn compressed_chunks_keep_arbitrary_canonical_bytes(bytes in proptest::collection::vec(proptest::prelude::any::<u8>(), 0..65536)) {
        if let Some(encoded) = codec::encode(&bytes).unwrap() {
            proptest::prop_assert!(encoded.len() < bytes.len());
            proptest::prop_assert_eq!(codec::decode(&encoded).unwrap(), bytes);
        }
    }

    #[test]
    fn repeated_patterns_roundtrip_with_bounded_compression(pattern in proptest::collection::vec(proptest::prelude::any::<u8>(), 1..256), repeats in 8usize..256) {
        let bytes = pattern.repeat(repeats);
        if let Some(encoded) = codec::encode(&bytes).unwrap() {
            proptest::prop_assert!(encoded.len() < bytes.len());
            proptest::prop_assert_eq!(codec::decode(&encoded).unwrap(), bytes);
        }
    }
}
