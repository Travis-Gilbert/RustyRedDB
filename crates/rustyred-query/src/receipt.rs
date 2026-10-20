use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryReceipt {
    pub receipt_digest: String,
    pub ast_hash: String,
    pub params_hash: String,
    pub snapshot_anchor: String,
    pub schema_digest: String,
    pub shape_key: String,
    pub selected_candidate_id: String,
    pub steering_reason: String,
    pub cost_units: f64,
    pub row_count: usize,
    pub duration_ms: u64,
    pub truncated: bool,
    pub tool: String,
    pub identity: Value,
}

impl QueryReceipt {
    /// Content identity for the receipt. Measurements (`cost_units`,
    /// `duration_ms`) stay attached and are not part of the hash, so the
    /// digest is recomputable for the same query over the same snapshot.
    pub fn digest_body(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
            self.ast_hash,
            self.params_hash,
            self.snapshot_anchor,
            self.schema_digest,
            self.shape_key,
            self.selected_candidate_id,
            self.steering_reason,
            self.row_count,
            self.truncated,
            self.tool
        )
    }

    pub fn finalize(mut self) -> Self {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(self.digest_body().as_bytes());
        self.receipt_digest = format!("sha256:{}", hex::encode(digest));
        self
    }
}

#[derive(Clone, Debug)]
pub struct MemoKey {
    pub ast_hash: String,
    pub params_hash: String,
    pub snapshot_anchor: String,
    pub schema_digest: String,
    pub row_cap: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample(duration_ms: u64, cost_units: f64) -> QueryReceipt {
        QueryReceipt {
            receipt_digest: String::new(),
            ast_hash: "sha256:ast".into(),
            params_hash: "sha256:params".into(),
            snapshot_anchor: "sha256:snap".into(),
            schema_digest: "sha256:schema".into(),
            shape_key: "node|Person".into(),
            selected_candidate_id: "node_scan".into(),
            steering_reason: "cold_start_native_floor".into(),
            cost_units,
            row_count: 1,
            duration_ms,
            truncated: false,
            tool: "cypher_query".into(),
            identity: json!({}),
        }
        .finalize()
    }

    #[test]
    fn receipt_digest_ignores_attached_measurements() {
        let fast = sample(4, 1.0);
        let slow = sample(40, 9.5);
        assert_eq!(fast.receipt_digest, slow.receipt_digest);
        assert_eq!(fast.digest_body(), slow.digest_body());
        assert_ne!(fast.duration_ms, slow.duration_ms);
        assert_ne!(fast.cost_units, slow.cost_units);
    }
}

impl MemoKey {
    pub fn cache_key(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}",
            self.ast_hash, self.params_hash, self.snapshot_anchor, self.schema_digest, self.row_cap
        )
    }
}

#[derive(Clone, Debug)]
pub struct MemoEntry {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
    pub receipt: QueryReceipt,
    pub plan: Value,
    pub total_available: usize,
}
