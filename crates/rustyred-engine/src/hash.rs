use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

use crate::graph::error::{GraphStoreError, GraphStoreResult};

const SHA256_PREFIX: &str = "sha256:";
const SHA256_TEXT_LEN: usize = SHA256_PREFIX.len() + 64;
const LOWER_HEX: &[u8; 16] = b"0123456789abcdef";

/// A canonical SHA-256 content address.
///
/// The bytes are deliberately private: callers must construct hashes from
/// content, exact digest bytes, or the strict prefixed wire representation.
/// There is no fabricated zero/default address.
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Hash([u8; 32]);

impl Hash {
    /// Hash `bytes` with SHA-256.
    pub fn of(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    /// Construct an address from its exact 32-byte digest.
    pub fn from_bytes(raw: [u8; 32]) -> Self {
        Self(raw)
    }

    /// Borrow the exact 32-byte digest.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Render the canonical `sha256:<lowercase hex>` wire representation.
    pub fn to_prefixed(&self) -> String {
        let mut output = String::with_capacity(SHA256_TEXT_LEN);
        output.push_str(SHA256_PREFIX);
        for byte in self.0 {
            output.push(LOWER_HEX[usize::from(byte >> 4)] as char);
            output.push(LOWER_HEX[usize::from(byte & 0x0f)] as char);
        }
        output
    }

    /// Parse the exact canonical `sha256:<64 lowercase hex>` representation.
    pub fn parse_prefixed(text: &str) -> GraphStoreResult<Self> {
        if !text.starts_with(SHA256_PREFIX) {
            return Err(invalid_hash(format!(
                "hash must start with the {SHA256_PREFIX} prefix"
            )));
        }
        if text.len() != SHA256_TEXT_LEN {
            return Err(invalid_hash(format!(
                "hash must contain exactly 64 lowercase hexadecimal digits after {SHA256_PREFIX}"
            )));
        }

        let encoded = &text.as_bytes()[SHA256_PREFIX.len()..];
        let mut raw = [0_u8; 32];
        for (index, pair) in encoded.chunks_exact(2).enumerate() {
            let high = decode_lower_hex(pair[0]).ok_or_else(|| {
                invalid_hash(format!(
                    "hash contains a non-lowercase-hexadecimal byte at offset {}",
                    SHA256_PREFIX.len() + index * 2
                ))
            })?;
            let low = decode_lower_hex(pair[1]).ok_or_else(|| {
                invalid_hash(format!(
                    "hash contains a non-lowercase-hexadecimal byte at offset {}",
                    SHA256_PREFIX.len() + index * 2 + 1
                ))
            })?;
            raw[index] = (high << 4) | low;
        }

        Ok(Self(raw))
    }

    /// Read the final two digest bytes as a little-endian boundary value.
    pub fn trailing_u16(&self) -> u16 {
        u16::from_le_bytes([self.0[30], self.0[31]])
    }
}

impl fmt::Display for Hash {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(SHA256_PREFIX)?;
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Hash {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Hash({self})")
    }
}

impl FromStr for Hash {
    type Err = GraphStoreError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::parse_prefixed(text)
    }
}

impl Serialize for Hash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_prefixed())
    }
}

impl<'de> Deserialize<'de> for Hash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        Self::parse_prefixed(&text)
            .map_err(|error| serde::de::Error::custom(format!("{}: {}", error.code, error.message)))
    }
}

fn decode_lower_hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn invalid_hash(message: impl Into<String>) -> GraphStoreError {
    GraphStoreError::new("invalid_hash", message)
}

#[cfg(test)]
mod tests {
    use super::Hash;

    const ABC_SHA256: &str =
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    #[test]
    fn sha256_abc_matches_the_golden_address_and_boundary() {
        let hash = Hash::of(b"abc");

        assert_eq!(hash.to_prefixed(), ABC_SHA256);
        assert_eq!(hash.trailing_u16(), 0xad15);
    }

    #[test]
    fn raw_string_and_serde_round_trip() {
        let original = Hash::of(b"typed content address");
        let raw = *original.as_bytes();
        let from_raw = Hash::from_bytes(raw);

        assert_eq!(from_raw, original);
        assert_eq!(original.to_string().parse::<Hash>().unwrap(), original);

        let json = serde_json::to_string(&original).unwrap();
        assert_eq!(json, format!("\"{}\"", original));
        assert_eq!(serde_json::from_str::<Hash>(&json).unwrap(), original);
    }

    #[test]
    fn debug_is_explicit_and_address_shaped() {
        let hash = Hash::of(b"abc");

        assert_eq!(format!("{hash:?}"), format!("Hash({ABC_SHA256})"));
    }

    #[test]
    fn strict_parser_rejects_noncanonical_addresses() {
        let uppercase_hex = ABC_SHA256.replace("ba7816", "BA7816");
        let wrong_prefix = ABC_SHA256.replacen("sha256:", "blake3:", 1);
        let too_short = &ABC_SHA256[..ABC_SHA256.len() - 1];
        let non_hex = ABC_SHA256.replacen("ba7816", "ga7816", 1);

        for malformed in [&uppercase_hex, &wrong_prefix, too_short, &non_hex] {
            let error = Hash::parse_prefixed(malformed).unwrap_err();
            assert_eq!(error.code, "invalid_hash");
        }
    }

    #[test]
    fn serde_rejects_non_string_and_noncanonical_values() {
        assert!(serde_json::from_str::<Hash>("null").is_err());
        assert!(
            serde_json::from_str::<Hash>(&format!("\"{}\"", ABC_SHA256.to_uppercase())).is_err()
        );
    }
}
