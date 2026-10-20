use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::sync::Arc;

use rustyred_thg_core::{
    ColdFragment, ColdIndex, DiskColdIndex, FoldId, FoldStorage, FoldSurface,
    MaterializationEligibility, MutationOrder, PartialFoldRuntime, PartialKey, PartialOperator,
    PositionedFoldIndex, RedbFoldStorage, ResolvedRow, ScalarValue,
};
use theorem_oplog::LogPosition;

#[test]
fn every_fold_exposes_indexed_partial_and_explicit_full_surfaces() {
    for fold_id in FoldId::ALL {
        let surfaces = fold_id.surfaces();
        assert!(
            surfaces.iter().all(|surface| surface.fold_id() == fold_id),
            "{fold_id} advertises a foreign surface"
        );
        assert!(surfaces.iter().any(|surface| matches!(
            surface.eligibility(),
            MaterializationEligibility::IndexedPartial(operator) if operator.fold_id() == fold_id
        )));
        assert!(surfaces
            .iter()
            .any(|surface| matches!(surface.eligibility(), MaterializationEligibility::FullState)));
    }
    assert_eq!(
        FoldSurface::GraphAllNodes.eligibility(),
        MaterializationEligibility::FullState
    );
    assert_eq!(
        FoldSurface::HistoryCheckout.eligibility(),
        MaterializationEligibility::FullState
    );
}

#[test]
fn partial_runtime_matches_full_positioned_oracle_across_four_folds() -> Result<(), Box<dyn Error>>
{
    let directory = tempfile::tempdir()?;
    let storage: Arc<dyn FoldStorage> =
        Arc::new(RedbFoldStorage::open(directory.path().join("floor"))?);
    let cold: Arc<dyn ColdIndex> = Arc::new(DiskColdIndex::open(directory.path().join("cold"))?);
    let index = PositionedFoldIndex::new(storage, Arc::clone(&cold));
    let mut runtime = PartialFoldRuntime::new(index.clone(), 24 * 1024)?;
    let operators = [
        PartialOperator::FactsCurrentEntity,
        PartialOperator::GraphNode,
        PartialOperator::AlgebraRegion,
        PartialOperator::HistoryInterval,
    ];
    let mut oracle = BTreeMap::<PartialKey, BTreeMap<MutationOrder, Option<Vec<u8>>>>::new();
    let mut random = DeterministicRandom::new(0xF01D_05D1_FF3A_3A71);

    for position in 1..=160_u64 {
        let operator = operators[random.usize(operators.len())];
        let key = format!("key:{:02}", random.usize(24)).into_bytes();
        let order = MutationOrder::new(LogPosition(position), 0)?;
        let payload = (random.usize(7) != 0)
            .then(|| format!("value:{position}:{}", random.next()).into_bytes());
        runtime.apply_update(operator, &key, order, payload.clone())?;
        oracle
            .entry(PartialKey::new(operator, key.clone())?)
            .or_default()
            .insert(order, payload);

        for _ in 0..4 {
            let query_operator = operators[random.usize(operators.len())];
            let query_key = format!("key:{:02}", random.usize(24)).into_bytes();
            let query_position = LogPosition(1 + random.usize(position as usize) as u64);
            let expected = oracle
                .get(&PartialKey::new(query_operator, query_key.clone())?)
                .and_then(|history| {
                    history
                        .iter()
                        .rev()
                        .find(|(order, _)| order.position <= query_position)
                        .map(|(_, value)| value.clone())
                })
                .flatten();
            let actual = runtime.read_at(query_operator, &query_key, query_position)?;
            assert_eq!(actual.payload, expected);
            assert_eq!(actual.through, query_position);
            assert!(runtime.resident_bytes() <= runtime.budget_bytes());
        }

        if position % 13 == 0 {
            let operator = operators[random.usize(operators.len())];
            let key = format!("key:{:02}", random.usize(24)).into_bytes();
            runtime.evict(operator, &key)?;
        }
    }

    let published_locators = oracle
        .iter()
        .flat_map(|(key, history)| {
            history.keys().map(|order| {
                index
                    .locator_id(key.operator, &key.key, *order)
                    .expect("oracle key is valid")
            })
        })
        .collect::<BTreeSet<_>>();
    assert!(published_locators
        .iter()
        .all(|id| cold.lookup(id).unwrap().is_some()));
    assert!(runtime.resident_len() < oracle.len());
    Ok(())
}

#[test]
fn cold_fragment_round_trips_through_positioned_reconstruction() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let storage: Arc<dyn FoldStorage> =
        Arc::new(RedbFoldStorage::open(directory.path().join("floor"))?);
    let cold: Arc<dyn ColdIndex> = Arc::new(DiskColdIndex::open(directory.path().join("cold"))?);
    let mut runtime = PartialFoldRuntime::new(PositionedFoldIndex::new(storage, cold), 16 * 1024)?;
    let fragment = ColdFragment::from_rows(
        "facts:entity:1",
        "facts",
        &[ResolvedRow::new(
            "facts",
            "entity:1",
            BTreeMap::from([
                (
                    "attribute".to_string(),
                    ScalarValue::String("name".to_string()),
                ),
                ("version".to_string(), ScalarValue::I64(7)),
            ]),
        )],
    );
    let bytes = fragment.reconstruction_bytes()?;
    runtime.apply_update(
        PartialOperator::FactsEntityAttribute,
        b"entity:1\0name",
        MutationOrder::new(LogPosition(7), 0)?,
        Some(bytes),
    )?;
    let read = runtime.read_at(
        PartialOperator::FactsEntityAttribute,
        b"entity:1\0name",
        LogPosition(7),
    )?;
    let decoded = ColdFragment::from_reconstruction_bytes(
        read.payload.as_deref().expect("published fragment value"),
    )?;
    assert_eq!(decoded, fragment);
    Ok(())
}

struct DeterministicRandom(u64);

impl DeterministicRandom {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn usize(&mut self, upper: usize) -> usize {
        (self.next() as usize) % upper
    }
}
