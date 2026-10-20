//! Per-node SoA state partition (SPEC-NODE-STATE-PARTITION minimal surface).
//!
//! Criticism-survival (SPEC-CRITICISM-SURVIVAL) writes acceptability into
//! [`NodeStateChannel::Corroboration`] and attack pressure into
//! [`NodeStateChannel::CriticismPressure`]. Additional channels land with the
//! full partition spec; this module is the shared slot-indexed store those
//! writers already need.

use serde::{Deserialize, Serialize};

/// Named f64 channel inside a [`NodeStateBlock`].
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeStateChannel {
    /// Acceptability Acc(c) from the weighted h-categoriser.
    Corroboration,
    /// Attack pressure Pressure(c) = Σ Acc(a) * severity(a → c).
    CriticismPressure,
}

impl NodeStateChannel {
    pub const ALL: [Self; 2] = [Self::Corroboration, Self::CriticismPressure];

    pub fn index(self) -> usize {
        match self {
            Self::Corroboration => 0,
            Self::CriticismPressure => 1,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Corroboration => "corroboration",
            Self::CriticismPressure => "criticism_pressure",
        }
    }
}

/// Slot-indexed SoA block of per-node epistemic channels.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NodeStateBlock {
    len: usize,
    channels: [Vec<f64>; 2],
}

impl NodeStateBlock {
    pub fn new(len: usize) -> Self {
        Self {
            len,
            channels: [vec![0.0; len], vec![0.0; len]],
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn get(&self, channel: NodeStateChannel, slot: u32) -> f64 {
        self.channels[channel.index()]
            .get(slot as usize)
            .copied()
            .unwrap_or(0.0)
    }

    pub fn set(&mut self, channel: NodeStateChannel, slot: u32, value: f64) {
        let idx = slot as usize;
        if idx >= self.len {
            return;
        }
        self.channels[channel.index()][idx] = value;
    }

    pub fn channel(&self, channel: NodeStateChannel) -> &[f64] {
        &self.channels[channel.index()]
    }

    pub fn channel_mut(&mut self, channel: NodeStateChannel) -> &mut [f64] {
        &mut self.channels[channel.index()]
    }

    pub fn fill(&mut self, channel: NodeStateChannel, value: f64) {
        for slot in self.channels[channel.index()].iter_mut() {
            *slot = value;
        }
    }

    pub fn resize(&mut self, len: usize) {
        self.len = len;
        for channel in &mut self.channels {
            channel.resize(len, 0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channels_are_independent() {
        let mut state = NodeStateBlock::new(3);
        state.set(NodeStateChannel::Corroboration, 1, 0.75);
        state.set(NodeStateChannel::CriticismPressure, 1, 0.25);
        assert!((state.get(NodeStateChannel::Corroboration, 1) - 0.75).abs() < 1e-12);
        assert!((state.get(NodeStateChannel::CriticismPressure, 1) - 0.25).abs() < 1e-12);
        assert_eq!(state.get(NodeStateChannel::Corroboration, 0), 0.0);
    }
}
