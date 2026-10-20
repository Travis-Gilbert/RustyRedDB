use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::access_method::{ColumnId, RowId, ScalarBound, ScalarValue};
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::relational::ResolvedRow;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressionFilter {
    Zstandard,
    DoubleDelta,
    RunLength,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ZoneMap {
    pub min: ScalarValue,
    pub max: ScalarValue,
}

impl ZoneMap {
    pub fn excludes_range(&self, lo: &ScalarBound, hi: &ScalarBound) -> bool {
        match (self.min.as_f64(), self.max.as_f64()) {
            (Some(min), Some(max)) => {
                let lo_above_max = match lo {
                    ScalarBound::Unbounded => false,
                    ScalarBound::Included(value) | ScalarBound::Excluded(value) => {
                        value.as_f64().map(|lo| lo > max).unwrap_or(false)
                    }
                };
                let hi_below_min = match hi {
                    ScalarBound::Unbounded => false,
                    ScalarBound::Included(value) | ScalarBound::Excluded(value) => {
                        value.as_f64().map(|hi| hi < min).unwrap_or(false)
                    }
                };
                lo_above_max || hi_below_min
            }
            _ => false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FragmentColumn {
    pub column: ColumnId,
    pub values: Vec<Option<ScalarValue>>,
    pub filters: Vec<CompressionFilter>,
    pub zone_map: Option<ZoneMap>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ColdFragment {
    pub fragment_id: String,
    pub relation: String,
    pub row_ids: Vec<RowId>,
    pub columns: BTreeMap<ColumnId, FragmentColumn>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct FragmentRangeStats {
    pub fragments_visited: usize,
    pub fragments_skipped: usize,
    pub rows_returned: usize,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct FragmentRangeResult {
    pub rows: Vec<RowId>,
    pub stats: FragmentRangeStats,
}

impl ColdFragment {
    pub fn from_rows(
        fragment_id: impl Into<String>,
        relation: impl Into<String>,
        rows: &[ResolvedRow],
    ) -> Self {
        let mut columns: BTreeMap<ColumnId, Vec<Option<ScalarValue>>> = BTreeMap::new();
        for row in rows {
            for column in row.values.keys() {
                columns.entry(column.clone()).or_default();
            }
        }
        for row in rows {
            for values in columns.values_mut() {
                values.push(None);
            }
            let idx = row_ids_len(values_len(&columns)).saturating_sub(1);
            for (column, value) in &row.values {
                if let Some(values) = columns.get_mut(column) {
                    values[idx] = Some(value.clone());
                }
            }
        }
        let columns = columns
            .into_iter()
            .map(|(column, values)| {
                let filters = filters_for_values(&values);
                let zone_map = zone_map_for_values(&values);
                (
                    column.clone(),
                    FragmentColumn {
                        column,
                        values,
                        filters,
                        zone_map,
                    },
                )
            })
            .collect();
        Self {
            fragment_id: fragment_id.into(),
            relation: relation.into(),
            row_ids: rows.iter().map(|row| row.id.clone()).collect(),
            columns,
        }
    }

    pub fn range_query(
        &self,
        column: &str,
        lo: ScalarBound,
        hi: ScalarBound,
    ) -> GraphStoreResult<FragmentRangeResult> {
        let Some(column) = self.columns.get(column) else {
            return Ok(FragmentRangeResult {
                rows: Vec::new(),
                stats: FragmentRangeStats {
                    fragments_visited: 0,
                    fragments_skipped: 1,
                    rows_returned: 0,
                },
            });
        };
        let has_value = column.values.iter().any(Option::is_some);
        let has_numeric = column
            .values
            .iter()
            .flatten()
            .any(|value| value.as_f64().is_some());
        if has_value && !has_numeric {
            return Err(GraphStoreError::new(
                "invalid_fragment_range_column",
                format!("fragment column {} is not numeric", column.column),
            ));
        }
        if column
            .zone_map
            .as_ref()
            .map(|zone| zone.excludes_range(&lo, &hi))
            .unwrap_or(false)
        {
            return Ok(FragmentRangeResult {
                rows: Vec::new(),
                stats: FragmentRangeStats {
                    fragments_visited: 0,
                    fragments_skipped: 1,
                    rows_returned: 0,
                },
            });
        }
        let mut rows = Vec::new();
        for (idx, value) in column.values.iter().enumerate() {
            let Some(value) = value.as_ref().and_then(ScalarValue::as_f64) else {
                continue;
            };
            if lo.contains_numeric(value, true) && hi.contains_numeric(value, false) {
                if let Some(row_id) = self.row_ids.get(idx) {
                    rows.push(row_id.clone());
                }
            }
        }
        Ok(FragmentRangeResult {
            stats: FragmentRangeStats {
                fragments_visited: 1,
                fragments_skipped: 0,
                rows_returned: rows.len(),
            },
            rows,
        })
    }

    /// Stable payload for a positioned partial-fold reconstruction index.
    pub fn reconstruction_bytes(&self) -> GraphStoreResult<Vec<u8>> {
        serde_json::to_vec(self).map_err(|error| {
            GraphStoreError::new("cold_fragment_codec", format!("encode fragment: {error}"))
        })
    }

    pub fn from_reconstruction_bytes(bytes: &[u8]) -> GraphStoreResult<Self> {
        serde_json::from_slice(bytes).map_err(|error| {
            GraphStoreError::new("cold_fragment_codec", format!("decode fragment: {error}"))
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct ColdFragmentStore {
    fragments: Vec<ColdFragment>,
}

impl ColdFragmentStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn append(&mut self, fragment: ColdFragment) {
        self.fragments.push(fragment);
    }

    pub fn range_query(
        &self,
        relation: &str,
        column: &str,
        lo: ScalarBound,
        hi: ScalarBound,
    ) -> GraphStoreResult<FragmentRangeResult> {
        let mut result = FragmentRangeResult::default();
        let matching = self
            .fragments
            .iter()
            .filter(|fragment| fragment.relation == relation)
            .collect::<Vec<_>>();
        if matching.is_empty()
            || matching
                .iter()
                .all(|fragment| !fragment.columns.contains_key(column))
        {
            return Err(GraphStoreError::new(
                "invalid_fragment_range_column",
                format!("relation {relation} has no fragment column {column}"),
            ));
        }
        for fragment in matching {
            let partial = fragment.range_query(column, lo.clone(), hi.clone())?;
            result.stats.fragments_visited += partial.stats.fragments_visited;
            result.stats.fragments_skipped += partial.stats.fragments_skipped;
            result.rows.extend(partial.rows);
        }
        result.stats.rows_returned = result.rows.len();
        Ok(result)
    }
}

pub const EPISTEMIC_COLUMN_GROUP: &str = "epistemic_hot_columns";
pub const EPISTEMIC_COLUMN_NAMES: [&str; 6] = [
    "epistemic_weight",
    "source_reliability",
    "justification_prior",
    "valid_from_ms",
    "acceptance_status",
    "superseded",
];

pub fn is_epistemic_column(column: &str) -> bool {
    EPISTEMIC_COLUMN_NAMES.contains(&column)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HotColumnEncoding {
    Dictionary,
    BitPacked,
    QuantizedU16,
    DeltaFramed,
    Raw,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct HotColumnGroup {
    pub group_id: String,
    pub relation: String,
    pub columns: Vec<ColumnId>,
    pub block_size: usize,
    pub row_count: usize,
    pub fragments: Vec<ColdFragment>,
    pub rowwise_property_bytes: usize,
    pub unencoded_columnar_bytes: usize,
    pub encoded_columnar_bytes: usize,
    pub encodings: BTreeMap<ColumnId, HotColumnEncoding>,
}

impl HotColumnGroup {
    pub fn epistemic(relation: impl Into<String>, rows: &[ResolvedRow]) -> Self {
        Self::from_rows(relation, rows, &EPISTEMIC_COLUMN_NAMES, 16)
    }

    pub fn from_rows(
        relation: impl Into<String>,
        rows: &[ResolvedRow],
        columns: &[&str],
        block_size: usize,
    ) -> Self {
        let relation = relation.into();
        let block_size = block_size.max(1);
        let columns = columns
            .iter()
            .map(|column| (*column).to_string())
            .collect::<Vec<_>>();
        let filtered_rows = rows
            .iter()
            .filter_map(|row| {
                let values = columns
                    .iter()
                    .filter_map(|column| {
                        row.values
                            .get(column)
                            .cloned()
                            .map(|value| (column.clone(), value))
                    })
                    .collect::<BTreeMap<_, _>>();
                if values.is_empty() {
                    return None;
                }
                let mut out = ResolvedRow::new(relation.clone(), row.id.clone(), values);
                out.properties = row.properties.clone();
                Some(out)
            })
            .collect::<Vec<_>>();
        let fragments = filtered_rows
            .chunks(block_size)
            .enumerate()
            .map(|(idx, chunk)| {
                ColdFragment::from_rows(
                    format!("hot:{}:{}:{idx}", EPISTEMIC_COLUMN_GROUP, relation),
                    relation.clone(),
                    chunk,
                )
            })
            .collect::<Vec<_>>();
        let encodings = columns
            .iter()
            .map(|column| (column.clone(), encoding_for_hot_column(column)))
            .collect::<BTreeMap<_, _>>();
        let rowwise_property_bytes = estimate_rowwise_property_bytes(rows, &columns);
        let unencoded_columnar_bytes = estimate_unencoded_columnar_bytes(&fragments);
        let encoded_columnar_bytes = estimate_encoded_columnar_bytes(&fragments, &encodings);
        Self {
            group_id: EPISTEMIC_COLUMN_GROUP.to_string(),
            relation,
            columns,
            block_size,
            row_count: filtered_rows.len(),
            fragments,
            rowwise_property_bytes,
            unencoded_columnar_bytes,
            encoded_columnar_bytes,
            encodings,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.row_count == 0
    }

    pub fn range_query(
        &self,
        column: &str,
        lo: ScalarBound,
        hi: ScalarBound,
    ) -> GraphStoreResult<FragmentRangeResult> {
        let mut result = FragmentRangeResult::default();
        if self.fragments.is_empty()
            || self
                .fragments
                .iter()
                .all(|fragment| !fragment.columns.contains_key(column))
        {
            return Err(GraphStoreError::new(
                "invalid_fragment_range_column",
                format!("hot column group {} has no column {column}", self.group_id),
            ));
        }
        for fragment in &self.fragments {
            let partial = fragment.range_query(column, lo.clone(), hi.clone())?;
            result.stats.fragments_visited += partial.stats.fragments_visited;
            result.stats.fragments_skipped += partial.stats.fragments_skipped;
            result.rows.extend(partial.rows);
        }
        result.stats.rows_returned = result.rows.len();
        Ok(result)
    }

    pub fn value(&self, row_id: &str, column: &str) -> Option<&ScalarValue> {
        for fragment in &self.fragments {
            let Some(row_index) = fragment
                .row_ids
                .iter()
                .position(|candidate| candidate == row_id)
            else {
                continue;
            };
            return fragment
                .columns
                .get(column)
                .and_then(|column| column.values.get(row_index))
                .and_then(Option::as_ref);
        }
        None
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PromotionPolicy {
    pub recall_threshold: u64,
    pub indexed_columns: Vec<ColumnId>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PromotionOutcome {
    pub promoted: bool,
    pub fragment_id: Option<String>,
    pub hot_handle: ResolvedRow,
}

impl PromotionPolicy {
    pub fn new(recall_threshold: u64, indexed_columns: Vec<ColumnId>) -> Self {
        Self {
            recall_threshold: recall_threshold.max(1),
            indexed_columns,
        }
    }

    pub fn promote_if_recalled(
        &self,
        row: &ResolvedRow,
        recall_count: u64,
        fragments: &mut ColdFragmentStore,
    ) -> PromotionOutcome {
        if recall_count < self.recall_threshold {
            return PromotionOutcome {
                promoted: false,
                fragment_id: None,
                hot_handle: row.clone(),
            };
        }
        let fragment_id = format!("cold:{}:{}", row.relation, row.id);
        fragments.append(ColdFragment::from_rows(
            fragment_id.clone(),
            row.relation.clone(),
            std::slice::from_ref(row),
        ));
        let mut values = BTreeMap::new();
        for column in &self.indexed_columns {
            if let Some(value) = row.values.get(column) {
                values.insert(column.clone(), value.clone());
            }
        }
        values.insert(
            "cold_fragment_id".to_string(),
            ScalarValue::String(fragment_id.clone()),
        );
        PromotionOutcome {
            promoted: true,
            fragment_id: Some(fragment_id),
            hot_handle: ResolvedRow {
                id: row.id.clone(),
                relation: row.relation.clone(),
                values,
                properties: row.properties.clone(),
            },
        }
    }
}

fn filters_for_values(values: &[Option<ScalarValue>]) -> Vec<CompressionFilter> {
    let mut has_text = false;
    let mut has_time = false;
    let mut low_cardinality = true;
    let mut distinct = Vec::<String>::new();
    for value in values.iter().flatten() {
        match value {
            ScalarValue::String(value) => {
                has_text = true;
                if !distinct.contains(value) {
                    distinct.push(value.clone());
                }
            }
            ScalarValue::I64(_) => has_time = true,
            ScalarValue::F64(_) => {}
            ScalarValue::Bool(value) => {
                let value = value.to_string();
                if !distinct.contains(&value) {
                    distinct.push(value);
                }
            }
        }
    }
    low_cardinality &= distinct.len() <= 8;
    let mut filters = Vec::new();
    if has_text {
        filters.push(CompressionFilter::Zstandard);
    }
    if has_time {
        filters.push(CompressionFilter::DoubleDelta);
    }
    if low_cardinality && !distinct.is_empty() {
        filters.push(CompressionFilter::RunLength);
    }
    if filters.is_empty() {
        filters.push(CompressionFilter::Zstandard);
    }
    filters
}

fn zone_map_for_values(values: &[Option<ScalarValue>]) -> Option<ZoneMap> {
    let nums = values
        .iter()
        .flatten()
        .filter_map(ScalarValue::as_f64)
        .collect::<Vec<_>>();
    if nums.is_empty() {
        return None;
    }
    let min = nums.iter().copied().fold(f64::INFINITY, f64::min);
    let max = nums.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Some(ZoneMap {
        min: ScalarValue::F64(min),
        max: ScalarValue::F64(max),
    })
}

fn encoding_for_hot_column(column: &str) -> HotColumnEncoding {
    match column {
        "acceptance_status" => HotColumnEncoding::Dictionary,
        "superseded" => HotColumnEncoding::BitPacked,
        "epistemic_weight" | "source_reliability" | "justification_prior" => {
            HotColumnEncoding::QuantizedU16
        }
        "valid_from_ms" => HotColumnEncoding::DeltaFramed,
        _ => HotColumnEncoding::Raw,
    }
}

fn estimate_rowwise_property_bytes(rows: &[ResolvedRow], columns: &[ColumnId]) -> usize {
    rows.iter()
        .map(|row| {
            columns
                .iter()
                .filter_map(|column| row.values.get(column).map(|value| (column, value)))
                .map(|(column, value)| column.len() + 24 + scalar_value_bytes(value))
                .sum::<usize>()
                + 16
        })
        .sum()
}

fn estimate_unencoded_columnar_bytes(fragments: &[ColdFragment]) -> usize {
    fragments
        .iter()
        .flat_map(|fragment| fragment.columns.values())
        .map(|column| {
            column
                .values
                .iter()
                .map(|value| value.as_ref().map(scalar_value_bytes).unwrap_or(0) + 1)
                .sum::<usize>()
                + 16
        })
        .sum()
}

fn estimate_encoded_columnar_bytes(
    fragments: &[ColdFragment],
    encodings: &BTreeMap<ColumnId, HotColumnEncoding>,
) -> usize {
    let mut by_column: BTreeMap<&str, Vec<Option<&ScalarValue>>> = BTreeMap::new();
    for fragment in fragments {
        for (column, values) in &fragment.columns {
            by_column
                .entry(column)
                .or_default()
                .extend(values.values.iter().map(Option::as_ref));
        }
    }
    by_column
        .into_iter()
        .map(|(column, values)| match encodings.get(column) {
            Some(HotColumnEncoding::Dictionary) => {
                let mut distinct = Vec::<&str>::new();
                for value in values.iter().flatten().filter_map(|value| value.as_str()) {
                    if !distinct.contains(&value) {
                        distinct.push(value);
                    }
                }
                distinct.iter().map(|value| value.len()).sum::<usize>() + values.len() * 2 + 8
            }
            Some(HotColumnEncoding::BitPacked) => values.len().div_ceil(8) + 8,
            Some(HotColumnEncoding::QuantizedU16) => values.iter().flatten().count() * 2 + 16,
            Some(HotColumnEncoding::DeltaFramed) => values.iter().flatten().count() * 4 + 16,
            _ => values
                .iter()
                .map(|value| value.map(scalar_value_bytes).unwrap_or(0) + 1)
                .sum::<usize>(),
        })
        .sum()
}

fn scalar_value_bytes(value: &ScalarValue) -> usize {
    match value {
        ScalarValue::String(value) => value.len(),
        ScalarValue::I64(_) | ScalarValue::F64(_) => 8,
        ScalarValue::Bool(_) => 1,
    }
}

fn values_len(columns: &BTreeMap<ColumnId, Vec<Option<ScalarValue>>>) -> usize {
    columns.values().next().map(Vec::len).unwrap_or(0)
}

fn row_ids_len(value: usize) -> usize {
    value
}

pub fn invalid_fragment(message: impl Into<String>) -> GraphStoreError {
    GraphStoreError::new("invalid_cold_fragment", message.into())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn row(id: &str, ts: i64, kind: &str) -> ResolvedRow {
        ResolvedRow::new(
            "memory",
            id,
            BTreeMap::from([
                ("t_ms".to_string(), ScalarValue::I64(ts)),
                ("kind".to_string(), ScalarValue::String(kind.to_string())),
            ]),
        )
    }

    #[test]
    fn cold_fragment_is_columnar_compressed_and_zone_map_pruned() {
        let rows = vec![row("m1", 10, "episode"), row("m2", 20, "episode")];
        let fragment = ColdFragment::from_rows("frag:1", "memory", &rows);
        let ts_column = fragment.columns.get("t_ms").unwrap();
        assert_eq!(ts_column.values.len(), 2);
        assert!(ts_column.filters.contains(&CompressionFilter::DoubleDelta));
        assert_eq!(
            fragment.columns.get("kind").unwrap().filters,
            vec![CompressionFilter::Zstandard, CompressionFilter::RunLength]
        );

        let missed = fragment
            .range_query(
                "t_ms",
                ScalarBound::Included(ScalarValue::I64(100)),
                ScalarBound::Included(ScalarValue::I64(200)),
            )
            .unwrap();
        assert_eq!(missed.stats.fragments_skipped, 1);
        assert!(missed.rows.is_empty());

        let hit = fragment
            .range_query(
                "t_ms",
                ScalarBound::Included(ScalarValue::I64(15)),
                ScalarBound::Included(ScalarValue::I64(25)),
            )
            .unwrap();
        assert_eq!(hit.stats.fragments_visited, 1);
        assert_eq!(hit.rows, vec!["m2".to_string()]);
    }

    #[test]
    fn fragment_store_accumulates_skip_stats() {
        let mut store = ColdFragmentStore::new();
        store.append(ColdFragment::from_rows(
            "frag:1",
            "memory",
            &[row("m1", 10, "episode")],
        ));
        store.append(ColdFragment::from_rows(
            "frag:2",
            "memory",
            &[row("m2", 100, "episode")],
        ));
        let result = store
            .range_query(
                "memory",
                "t_ms",
                ScalarBound::Included(ScalarValue::I64(5)),
                ScalarBound::Included(ScalarValue::I64(15)),
            )
            .unwrap();
        assert_eq!(result.rows, vec!["m1".to_string()]);
        assert_eq!(result.stats.fragments_visited, 1);
        assert_eq!(result.stats.fragments_skipped, 1);
    }

    #[test]
    fn promotion_keeps_indexed_hot_handle_after_cold_fragment_write() {
        let row = row("m1", 10, "episode");
        let mut fragments = ColdFragmentStore::new();
        let policy = PromotionPolicy::new(2, vec!["t_ms".to_string(), "kind".to_string()]);

        let skipped = policy.promote_if_recalled(&row, 1, &mut fragments);
        assert!(!skipped.promoted);

        let promoted = policy.promote_if_recalled(&row, 2, &mut fragments);
        assert!(promoted.promoted);
        assert_eq!(promoted.fragment_id.as_deref(), Some("cold:memory:m1"));
        assert_eq!(
            promoted.hot_handle.values.get("kind"),
            Some(&ScalarValue::String("episode".to_string()))
        );
        assert_eq!(
            promoted.hot_handle.values.get("cold_fragment_id"),
            Some(&ScalarValue::String("cold:memory:m1".to_string()))
        );

        let lookup = fragments
            .range_query(
                "memory",
                "t_ms",
                ScalarBound::Included(ScalarValue::I64(5)),
                ScalarBound::Included(ScalarValue::I64(15)),
            )
            .unwrap();
        assert_eq!(lookup.rows, vec!["m1".to_string()]);
    }
}
