//! TurboQuant token encoding spawned from a [`CalibrationEpoch`].
//!
//! Exact f32 vectors never enter the log. This module produces packed codes
//! and scales under a named calibration epoch, which the Algebra fold then
//! stores as token-region state.

use crate::fold::algebra::{CalibrationEpoch, TOKEN_BIT_WIDTH, TURBOVEC_ENCODE_VERSION};
use crate::graph::record::{NodeRecord, CALIBRATION_EPOCH_LABEL};
use crate::Hash;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;
use turbovec::{CalibrationState, TurboQuantIndex};

/// An empty, calibrated `TurboQuantIndex` spawned from an epoch through
/// `from_parts` with zero rows.
pub struct CalibratedEncoder {
    index: TurboQuantIndex,
    epoch: String,
    dim: usize,
    bit_width: u8,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TokenEncodeError {
    #[error("{0}")]
    Invalid(String),
}

impl CalibratedEncoder {
    pub fn from_epoch(epoch: &CalibrationEpoch) -> Result<Self, TokenEncodeError> {
        if epoch.turbovec_version != TURBOVEC_ENCODE_VERSION {
            return Err(TokenEncodeError::Invalid(format!(
                "epoch turbovec_version {} does not match {TURBOVEC_ENCODE_VERSION}",
                epoch.turbovec_version
            )));
        }
        let index = TurboQuantIndex::from_parts(
            Some(epoch.dim),
            usize::from(epoch.bit_width),
            0,
            Vec::new(),
            Vec::new(),
            epoch.tqplus_shift.clone(),
            epoch.tqplus_scale.clone(),
        )
        .map_err(|error| TokenEncodeError::Invalid(error.to_string()))?;
        Ok(Self {
            index,
            epoch: epoch.id.clone(),
            dim: epoch.dim,
            bit_width: epoch.bit_width,
        })
    }

    pub fn calibration_state(&self) -> CalibrationState {
        self.index.calibration_state()
    }

    pub fn epoch(&self) -> &str {
        &self.epoch
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }

    /// Encode a row-major `vector_count * dim` matrix and return the packed
    /// codes and scales for the newly added rows.
    pub fn encode(
        &mut self,
        vectors: &[f32],
        dim: usize,
    ) -> Result<EncodedTokens, TokenEncodeError> {
        if dim != self.dim {
            return Err(TokenEncodeError::Invalid(format!(
                "encoder dim {} does not match input dim {dim}",
                self.dim
            )));
        }
        if vectors.len() % dim != 0 {
            return Err(TokenEncodeError::Invalid(
                "token encode buffer is not a multiple of dim".to_string(),
            ));
        }
        if vectors.iter().any(|value| !value.is_finite()) {
            return Err(TokenEncodeError::Invalid(
                "token encode buffer contains a non-finite coordinate".to_string(),
            ));
        }
        let before = self.index.len();
        self.index
            .add_2d(vectors, dim)
            .map_err(|error| TokenEncodeError::Invalid(error.to_string()))?;
        let added = self.index.len().saturating_sub(before);
        let vector_count = u32::try_from(added).map_err(|_| {
            TokenEncodeError::Invalid("encoded token count exceeds u32".to_string())
        })?;
        let bytes_per = packed_bytes_per_vector(dim, self.bit_width)?;
        let packed = self.index.packed_codes();
        let scales = self.index.scales();
        Ok(EncodedTokens {
            vector_count,
            packed_codes: packed[before * bytes_per..].to_vec(),
            scales: scales[before..].to_vec(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EncodedTokens {
    pub vector_count: u32,
    pub packed_codes: Vec<u8>,
    pub scales: Vec<f32>,
}

pub fn packed_bytes_per_vector(dim: usize, bit_width: u8) -> Result<usize, TokenEncodeError> {
    dim.checked_mul(usize::from(bit_width))
        .and_then(|bits| bits.checked_div(8))
        .ok_or_else(|| TokenEncodeError::Invalid("token packed-code size overflow".to_string()))
}

impl CalibrationEpoch {
    /// `calibrate_2d(sample, dim)` on a fresh index, read the pair, address
    /// the sample and the body.
    pub fn fit(
        sample: &[f32],
        dim: usize,
        bit_width: u8,
        model_id: &str,
        model_version: &str,
        sample_block: String,
    ) -> Result<Self, TokenEncodeError> {
        if dim == 0 || dim % 8 != 0 {
            return Err(TokenEncodeError::Invalid(
                "calibration dim must be a positive multiple of 8".to_string(),
            ));
        }
        if !(2..=4).contains(&bit_width) {
            return Err(TokenEncodeError::Invalid(
                "calibration bit_width must be in 2..=4".to_string(),
            ));
        }
        if sample.len() % dim != 0 {
            return Err(TokenEncodeError::Invalid(
                "calibration sample is not a multiple of dim".to_string(),
            ));
        }
        let rows = sample.len() / dim;
        if rows < turbovec::MIN_CALIBRATION_ROWS {
            return Err(TokenEncodeError::Invalid(format!(
                "calibration sample has {rows} rows; minimum is {}",
                turbovec::MIN_CALIBRATION_ROWS
            )));
        }
        let mut index = TurboQuantIndex::new(dim, usize::from(bit_width))
            .map_err(|error| TokenEncodeError::Invalid(error.to_string()))?;
        index
            .calibrate_2d(sample, dim)
            .map_err(|error| TokenEncodeError::Invalid(error.to_string()))?;
        let mut epoch = Self {
            id: String::new(),
            node_id: String::new(),
            model_id: model_id.to_string(),
            model_version: model_version.to_string(),
            dim,
            bit_width,
            turbovec_version: TURBOVEC_ENCODE_VERSION.to_string(),
            sample_block,
            tqplus_shift: index.tqplus_shift().to_vec(),
            tqplus_scale: index.tqplus_scale().to_vec(),
            active: true,
        };
        epoch.id = epoch.content_id();
        epoch.node_id = format!("calibration-epoch:{}", epoch.id);
        Ok(epoch)
    }

    pub fn addressed_body(&self) -> Vec<u8> {
        canonical_epoch_body(self)
    }

    pub fn content_id(&self) -> String {
        Hash::of(&canonical_epoch_body(self)).to_prefixed()
    }

    pub fn to_node(&self) -> NodeRecord {
        NodeRecord::new(
            self.node_id.clone(),
            [CALIBRATION_EPOCH_LABEL],
            json!({
                "id": self.id,
                "model_id": self.model_id,
                "model_version": self.model_version,
                "dim": self.dim,
                "bit_width": self.bit_width,
                "turbovec_version": self.turbovec_version,
                "sample_block": self.sample_block,
                "tqplus_shift": self.tqplus_shift,
                "tqplus_scale": self.tqplus_scale,
                "active": self.active,
            }),
        )
    }

    pub fn from_node(node: &NodeRecord) -> Result<Self, crate::fold_quarantine::FoldConsumeError> {
        use crate::fold_quarantine::{FoldConsumeError, FoldRefusalCode};
        let schema =
            |reason: String| FoldConsumeError::refused(FoldRefusalCode::SchemaViolation, reason);
        if !node
            .labels
            .iter()
            .any(|label| label == CALIBRATION_EPOCH_LABEL)
        {
            return Err(schema(
                "CalibrationEpoch::from_node requires the CalibrationEpoch label".to_string(),
            ));
        }
        let properties = node
            .properties
            .as_object()
            .ok_or_else(|| schema("CalibrationEpoch properties must be an object".to_string()))?;
        let require_str = |key: &str| -> Result<String, crate::fold_quarantine::FoldConsumeError> {
            properties
                .get(key)
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| schema(format!("CalibrationEpoch missing string {key}")))
        };
        let dim = usize::try_from(
            properties
                .get("dim")
                .and_then(Value::as_u64)
                .ok_or_else(|| schema("CalibrationEpoch missing dim".to_string()))?,
        )
        .map_err(|_| schema("CalibrationEpoch dim exceeds usize".to_string()))?;
        let bit_width = u8::try_from(
            properties
                .get("bit_width")
                .and_then(Value::as_u64)
                .ok_or_else(|| schema("CalibrationEpoch missing bit_width".to_string()))?,
        )
        .map_err(|_| schema("CalibrationEpoch bit_width exceeds u8".to_string()))?;
        let tqplus_shift =
            f32_vec(properties.get("tqplus_shift"), "tqplus_shift").map_err(schema)?;
        let tqplus_scale =
            f32_vec(properties.get("tqplus_scale"), "tqplus_scale").map_err(schema)?;
        let epoch = Self {
            id: require_str("id")?,
            node_id: node.id.clone(),
            model_id: require_str("model_id")?,
            model_version: require_str("model_version")?,
            dim,
            bit_width,
            turbovec_version: require_str("turbovec_version")?,
            sample_block: require_str("sample_block")?,
            tqplus_shift,
            tqplus_scale,
            active: properties
                .get("active")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        };
        if epoch.turbovec_version != TURBOVEC_ENCODE_VERSION {
            return Err(schema(format!(
                "CalibrationEpoch turbovec_version {} is not {TURBOVEC_ENCODE_VERSION}",
                epoch.turbovec_version
            )));
        }
        let canonical_id = epoch.content_id();
        if epoch.id != canonical_id {
            return Err(schema(format!(
                "CalibrationEpoch id {} does not match canonical content id {canonical_id}",
                epoch.id
            )));
        }
        let canonical_node_id = format!("calibration-epoch:{canonical_id}");
        if epoch.node_id != canonical_node_id {
            return Err(schema(format!(
                "CalibrationEpoch node id {} does not match canonical node id {canonical_node_id}",
                epoch.node_id
            )));
        }
        Ok(epoch)
    }
}

fn canonical_epoch_body(epoch: &CalibrationEpoch) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "model_id": epoch.model_id,
        "model_version": epoch.model_version,
        "dim": epoch.dim,
        "bit_width": epoch.bit_width,
        "turbovec_version": epoch.turbovec_version,
        "sample_block": epoch.sample_block,
        "tqplus_shift": epoch.tqplus_shift,
        "tqplus_scale": epoch.tqplus_scale,
    }))
    .expect("calibration epoch body is json")
}

fn f32_vec(value: Option<&Value>, field: &str) -> Result<Vec<f32>, String> {
    let array = value
        .and_then(Value::as_array)
        .ok_or_else(|| format!("CalibrationEpoch missing {field}"))?;
    array
        .iter()
        .map(|item| {
            let number = item
                .as_f64()
                .ok_or_else(|| format!("CalibrationEpoch {field} must contain f32 values"))?;
            if !number.is_finite() || number < f32::MIN as f64 || number > f32::MAX as f64 {
                return Err(format!(
                    "CalibrationEpoch {field} contains a value outside the finite f32 range"
                ));
            }
            Ok(number as f32)
        })
        .collect()
}

pub fn default_token_bit_width() -> u8 {
    TOKEN_BIT_WIDTH
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_sample(rows: usize, dim: usize) -> Vec<f32> {
        let mut sample = Vec::with_capacity(rows * dim);
        for row in 0..rows {
            for col in 0..dim {
                let value = (((row * 31 + col * 17) % 97) as f32) / 97.0 - 0.5;
                sample.push(value);
            }
        }
        sample
    }

    #[test]
    fn encoder_from_fit_is_calibrated_and_deterministic() {
        let dim = 128;
        let sample = unit_sample(8, dim);
        let epoch = CalibrationEpoch::fit(
            &sample,
            dim,
            TOKEN_BIT_WIDTH,
            "fixture-model",
            "2026-08",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
        )
        .expect("fit");
        assert_eq!(epoch.tqplus_shift.len(), dim);
        let mut left = CalibratedEncoder::from_epoch(&epoch).expect("encoder");
        let mut right = CalibratedEncoder::from_epoch(&epoch).expect("encoder");
        assert_eq!(left.calibration_state(), CalibrationState::Calibrated);
        let set = unit_sample(4, dim);
        let a = left.encode(&set, dim).expect("encode a");
        let b = left.encode(&set, dim).expect("encode a again");
        let c = right.encode(&set, dim).expect("encode other encoder");
        assert_eq!(a.packed_codes, b.packed_codes);
        assert_eq!(a.scales, b.scales);
        assert_eq!(a.packed_codes, c.packed_codes);
        assert_eq!(a.scales, c.scales);
        let mut via_fit = TurboQuantIndex::new(dim, usize::from(TOKEN_BIT_WIDTH)).expect("index");
        via_fit.calibrate_2d(&sample, dim).expect("calibrate");
        via_fit.add_2d(&set, dim).expect("add");
        assert_eq!(
            a.packed_codes,
            via_fit.packed_codes()[..a.packed_codes.len()]
        );
    }
}
