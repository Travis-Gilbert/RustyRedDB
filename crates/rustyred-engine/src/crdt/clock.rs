use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

use crate::graph::ttl::now_ms;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ActorId([u8; 16]);

impl ActorId {
    pub const ZERO: Self = Self([0; 16]);

    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub fn from_label(label: &str) -> Self {
        let trimmed = label.trim();
        if trimmed.is_empty() {
            return Self::ZERO;
        }
        let digest = Sha256::digest(trimmed.as_bytes());
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&digest[..16]);
        Self(bytes)
    }

    pub fn to_hex(self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

impl From<&str> for ActorId {
    fn from(value: &str) -> Self {
        Self::from_label(value)
    }
}

impl From<String> for ActorId {
    fn from(value: String) -> Self {
        Self::from_label(&value)
    }
}

impl std::fmt::Display for ActorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl Serialize for ActorId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for ActorId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let trimmed = raw.trim();
        if trimmed.len() == 32 && trimmed.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            let mut bytes = [0u8; 16];
            for (idx, chunk) in trimmed.as_bytes().chunks_exact(2).enumerate() {
                let text = std::str::from_utf8(chunk).map_err(serde::de::Error::custom)?;
                bytes[idx] = u8::from_str_radix(text, 16).map_err(serde::de::Error::custom)?;
            }
            Ok(Self(bytes))
        } else {
            Ok(Self::from_label(trimmed))
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Hlc {
    pub physical_ms: i64,
    pub logical: u32,
    pub actor: ActorId,
}

impl Hlc {
    pub fn new(physical_ms: i64, logical: u32, actor: ActorId) -> Self {
        Self {
            physical_ms,
            logical,
            actor,
        }
    }
}

impl Default for Hlc {
    fn default() -> Self {
        Self {
            physical_ms: 0,
            logical: 0,
            actor: ActorId::ZERO,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HlcClock {
    actor: ActorId,
    last: Hlc,
}

impl HlcClock {
    pub fn new(actor: impl Into<ActorId>) -> Self {
        let actor = actor.into();
        Self {
            actor,
            last: Hlc::new(0, 0, actor),
        }
    }

    pub fn with_last(actor: impl Into<ActorId>, last: Hlc) -> Self {
        Self {
            actor: actor.into(),
            last,
        }
    }

    pub fn last(&self) -> Hlc {
        self.last
    }

    pub fn now(&mut self) -> Hlc {
        let physical = now_ms();
        let logical = if physical > self.last.physical_ms {
            0
        } else {
            self.last.logical.saturating_add(1)
        };
        self.last = Hlc::new(physical.max(self.last.physical_ms), logical, self.actor);
        self.last
    }

    pub fn observe(&mut self, remote: Hlc) -> Hlc {
        let physical = now_ms().max(self.last.physical_ms).max(remote.physical_ms);
        let logical = if physical == self.last.physical_ms && physical == remote.physical_ms {
            self.last.logical.max(remote.logical).saturating_add(1)
        } else if physical == self.last.physical_ms {
            self.last.logical.saturating_add(1)
        } else if physical == remote.physical_ms {
            remote.logical.saturating_add(1)
        } else {
            0
        };
        self.last = Hlc::new(physical, logical, self.actor);
        self.last
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_is_strictly_monotonic_for_same_actor() {
        let mut clock = HlcClock::new("codex");
        let first = clock.now();
        let second = clock.now();

        assert!(second > first);
    }

    #[test]
    fn observe_advances_beyond_local_and_remote() {
        let actor = ActorId::from_label("codex");
        let remote_actor = ActorId::from_label("claude-code");
        let local_last = Hlc::new(10, 2, actor);
        let mut clock = HlcClock::with_last(actor, local_last);
        let remote = Hlc::new(100, 7, remote_actor);

        let observed = clock.observe(remote);

        assert!(observed > local_last);
        assert!(observed > remote);

        let behind = Hlc::new(50, 9, remote_actor);
        let advanced = clock.observe(behind);
        assert!(advanced > observed);
        assert!(advanced > behind);
    }

    #[test]
    fn actor_breaks_equal_physical_and_logical_ties() {
        let codex = Hlc::new(42, 3, ActorId::from_label("codex"));
        let claude = Hlc::new(42, 3, ActorId::from_label("claude-code"));

        assert_ne!(codex.actor, claude.actor);
        assert_ne!(codex.cmp(&claude), std::cmp::Ordering::Equal);
    }
}
