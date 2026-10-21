//! V2 physical chunk envelope. Canonical bytes and addresses never change.
//! Stored encoded bytes and admitted decoded bytes retain the journal bound.
use super::{ensure_payload_bound, GraphStoreError, GraphStoreResult, Hash};

const PREFIX_BYTES: usize = 4 + 32;

#[cfg(test)]
std::thread_local! {
    pub(super) static DECODE_CAPACITY: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

fn invalid(message: impl Into<String>) -> GraphStoreError {
    GraphStoreError::new("chunk_store_browser_compression_invalid", message)
}

/// Returns a compressed envelope only when it saves physical bytes. A failed
/// compressor refuses the write; it never substitutes different source bytes.
pub(super) fn encode(bytes: &[u8]) -> GraphStoreResult<Option<Vec<u8>>> {
    ensure_payload_bound(bytes.len())?;
    let compressed = zstd::bulk::compress(bytes, 1).map_err(|error| invalid(error.to_string()))?;
    if compressed.len().saturating_add(PREFIX_BYTES) >= bytes.len() {
        return Ok(None);
    }
    let mut envelope = Vec::with_capacity(PREFIX_BYTES + compressed.len());
    envelope.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    envelope.extend_from_slice(Hash::of(bytes).as_bytes());
    envelope.extend_from_slice(&compressed);
    ensure_payload_bound(envelope.len())?;
    Ok(Some(envelope))
}

pub(super) fn decoded_len(envelope: &[u8]) -> GraphStoreResult<u32> {
    ensure_payload_bound(envelope.len())?;
    if envelope.len() <= PREFIX_BYTES {
        return Err(invalid("compressed chunk envelope is truncated"));
    }
    let len = u32::from_le_bytes(envelope[..4].try_into().expect("checked prefix"));
    ensure_payload_bound(len as usize)?;
    // The writer emits compression only when it wins. This also bounds encoded
    // allocation by decoded allocation, independent of adversarial metadata.
    if envelope.len() >= len as usize {
        return Err(invalid("compressed chunk does not reduce physical bytes"));
    }
    Ok(len)
}

pub(super) fn decode(envelope: &[u8]) -> GraphStoreResult<Vec<u8>> {
    let len = decoded_len(envelope)? as usize;
    #[cfg(test)]
    DECODE_CAPACITY.with(|capacity| capacity.set(Some(len)));
    let bytes = zstd::bulk::decompress(&envelope[PREFIX_BYTES..], len)
        .map_err(|error| invalid(error.to_string()))?;
    if bytes.len() != len || Hash::of(&bytes).as_bytes().as_slice() != &envelope[4..PREFIX_BYTES] {
        return Err(invalid(
            "compressed chunk length or canonical hash mismatch",
        ));
    }
    Ok(bytes)
}
