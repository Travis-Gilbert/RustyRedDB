#![cfg(feature = "graphblas")]

use rustyred_thg_core::fold::algebra::{AlgebraFold, AlgebraReachability};
use rustyred_thg_core::{
    CheckpointRejectCode, EdgeRecord, Fold, FoldCapabilities, FoldConsumeError, FoldDescriptor,
    FoldError, FoldGraphStore, FoldId, FoldRuntime, FoldVersion, GraphSnapshot,
    GraphSnapshotSource, GraphWrite, NodeRecord, RestoreDisposition,
};
use rustyred_thg_derivation::AlgebraClass;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use tempfile::TempDir;
use theorem_oplog::{
    AccessPath, IntentKind, LogIntent, LogPosition, LogRecord, OpDigest, PhysicalLog, WriteSet,
};

const SEED: u64 = 0xA16E_B2A5_2026_0813;
const LINK_TYPE: &str = "LINKS";

struct Fixture {
    _directory: TempDir,
    log: PhysicalLog,
    snapshots: Vec<GraphSnapshot>,
}

#[derive(Default)]
struct AlternateProfileAlgebraFold(AlgebraFold);

impl Fold for AlternateProfileAlgebraFold {
    fn id(&self) -> FoldId {
        self.0.id()
    }

    fn version(&self) -> FoldVersion {
        self.0.version()
    }

    fn capabilities(&self) -> FoldCapabilities {
        self.0.capabilities()
    }

    fn descriptor(&self) -> FoldDescriptor {
        let mut descriptor = self.0.descriptor();
        descriptor
            .valuation_profile
            .as_mut()
            .expect("Algebra declares a valuation profile")
            .operations[0]
            .graphblas_operator_pair
            .multiply_operator = "first_bool".to_string();
        descriptor
    }

    fn apply(&mut self, entry: &LogRecord) -> Result<(), FoldConsumeError> {
        self.0.apply(entry)
    }

    fn query(&self, query: &Value, position: LogPosition) -> Result<Value, FoldError> {
        self.0.query(query, position)
    }

    fn upquery(
        &mut self,
        key: &Value,
        position: LogPosition,
        pinned_records: &[LogRecord],
    ) -> Result<Value, FoldError> {
        self.0.upquery(key, position, pinned_records)
    }

    fn checkpoint(&self) -> Result<Vec<u8>, FoldError> {
        self.0.checkpoint()
    }

    fn restore(&mut self, checkpoint: &[u8]) -> Result<(), FoldError> {
        self.0.restore(checkpoint)
    }

    fn reset(&mut self) -> Result<(), FoldError> {
        self.0.reset()
    }
}

#[test]
fn descriptor_separates_graphblas_pair_from_mathematical_semiring() {
    let descriptor = AlgebraFold::default().descriptor();
    assert_eq!(descriptor, rustyred_thg_core::fold::algebra::descriptor());
    assert_eq!(descriptor.fold_version.get(), 2);
    let profile = descriptor
        .valuation_profile
        .expect("Algebra fold declares its valuation profile");
    assert_eq!(profile.engine, "suitesparse_graphblas");
    assert_eq!(profile.operations.len(), 1);
    let operation = &profile.operations[0];
    assert_eq!(operation.operation_id, "reachability");
    assert_eq!(
        operation.graphblas_operator_pair.law_identity.class,
        AlgebraClass::GraphBlasOperatorPair
    );
    assert!(!operation
        .graphblas_operator_pair
        .law_identity
        .laws
        .is_mathematical_semiring());
    let mathematical = operation
        .mathematical_semiring
        .as_ref()
        .expect("boolean LOR.LAND has an independent V01 semiring certificate");
    assert_eq!(mathematical.class, AlgebraClass::MathematicalSemiring);
    assert!(mathematical.laws.is_mathematical_semiring());
}

#[test]
fn native_incremental_regions_match_full_recomputation() -> Result<(), Box<dyn Error>> {
    let fixture = build_fixture(96)?;
    let cut = LogPosition(18);
    let latest = fixture.log.latest_position();
    let query = reachable_query("node:0");
    let mut algebra = FoldRuntime::for_log(AlgebraFold::default(), &fixture.log)?;

    algebra.apply_range(&fixture.log, cut)?;
    assert_eq!(algebra.inner().residency()?.regions, 0);
    assert_eq!(algebra.inner().residency()?.matrix_entries, 0);
    assert_eq!(algebra.inner().residency()?.matrix_bytes, 0);
    assert_eq!(algebra.inner().stats().native_full_builds, 0);
    assert_eq!(algebra.inner().deployment_id(), Some(&deployment_id()));

    let answer = decode_answer(algebra.upquery(&fixture.log, &query, cut)?)?;
    assert_eq!(answer.reachable, oracle_at(&fixture, cut, "node:0"));
    assert_eq!(algebra.inner().residency()?.regions, 1);
    assert!(algebra.inner().residency()?.matrix_bytes > 0);
    assert_eq!(algebra.inner().stats().native_full_builds, 1);

    for position in (cut.0 + 1)..=latest.0 {
        let position = LogPosition(position);
        algebra.apply_range(&fixture.log, position)?;
        let answer = decode_answer(algebra.query(&query, position)?)?;
        assert_eq!(
            answer.reachable,
            oracle_at(&fixture, position, "node:0"),
            "native incremental mismatch at log position {position}"
        );
        assert_eq!(algebra.inner().stats().native_full_builds, 1);
    }
    assert!(algebra.inner().stats().native_delta_updates > 0);
    assert_eq!(algebra.inner().stats().interval_batches, latest.0);
    assert!(algebra.query(&query, cut).is_err());
    let pinned = decode_answer(algebra.upquery(&fixture.log, &query, cut)?)?;
    assert_eq!(pinned.reachable, oracle_at(&fixture, cut, "node:0"));

    let checkpoint = algebra.checkpoint(&fixture.log)?;
    let mut restored = FoldRuntime::for_log(AlgebraFold::default(), &fixture.log)?;
    assert!(matches!(
        restored.restore_or_rebuild(&checkpoint, &fixture.log)?,
        RestoreDisposition::Restored { .. }
    ));
    assert_eq!(restored.inner().residency()?.regions, 0);
    assert_eq!(restored.inner().residency()?.matrix_entries, 0);
    assert_eq!(restored.inner().residency()?.matrix_bytes, 0);
    let restored_answer = decode_answer(restored.upquery(&fixture.log, &query, latest)?)?;
    assert_eq!(
        restored_answer.reachable,
        oracle_at(&fixture, latest, "node:0")
    );

    let mut fold = algebra.into_inner();
    assert!(fold.evict_region(LINK_TYPE));
    assert_eq!(fold.residency()?.matrix_entries, 0);
    let rebuilt = decode_answer(fold.upquery(&query, latest, fixture.log.records())?)?;
    assert_eq!(rebuilt.reachable, oracle_at(&fixture, latest, "node:0"));
    assert_eq!(fold.residency()?.regions, 1);
    assert_eq!(fold.stats().native_full_builds, 2);
    Ok(())
}

#[test]
fn validly_hashed_checkpoint_refuses_operator_profile_mismatch() -> Result<(), Box<dyn Error>> {
    let fixture = build_fixture(12)?;
    let mut source = FoldRuntime::for_log(AlternateProfileAlgebraFold::default(), &fixture.log)?;
    source.apply_all(&fixture.log)?;
    let checkpoint = source.checkpoint(&fixture.log)?;
    assert_eq!(
        checkpoint
            .fold_descriptor
            .valuation_profile
            .as_ref()
            .expect("Algebra checkpoint has a valuation profile")
            .operations[0]
            .graphblas_operator_pair
            .multiply_operator,
        "first_bool"
    );
    let persisted = serde_json::to_vec(&checkpoint)?;
    let checkpoint = serde_json::from_slice(&persisted)?;

    let mut target = FoldRuntime::for_log(AlgebraFold::default(), &fixture.log)?;
    let RestoreDisposition::Rebuilt { rejection, replay } =
        target.restore_or_rebuild(&checkpoint, &fixture.log)?
    else {
        panic!("operator profile mismatch must rebuild")
    };
    assert_eq!(rejection.code, CheckpointRejectCode::FoldCapabilities);
    assert_eq!(replay.through, fixture.log.latest_position());
    assert_eq!(target.inner().residency()?.matrix_bytes, 0);
    Ok(())
}

#[test]
fn native_materialization_does_not_change_checkpoint_bytes() -> Result<(), Box<dyn Error>> {
    let fixture = build_fixture(12)?;
    let latest = fixture.log.latest_position();
    let mut algebra = FoldRuntime::for_log(AlgebraFold::default(), &fixture.log)?;
    algebra.apply_all(&fixture.log)?;
    let logical_checkpoint = algebra.checkpoint(&fixture.log)?;
    assert_eq!(algebra.inner().residency()?.matrix_bytes, 0);

    algebra.upquery(&fixture.log, &reachable_query("node:0"), latest)?;
    assert!(algebra.inner().residency()?.matrix_bytes > 0);
    let materialized_checkpoint = algebra.checkpoint(&fixture.log)?;

    assert_eq!(materialized_checkpoint, logical_checkpoint);
    Ok(())
}

#[test]
fn parallel_edge_deletion_preserves_boolean_adjacency() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let mut store = FoldGraphStore::open(directory.path(), deployment_id())?;
    store.commit_batch(rustyred_thg_core::GraphMutationBatch::new([
        rustyred_thg_core::GraphMutation::NodeUpsert(node("node:a")),
        rustyred_thg_core::GraphMutation::NodeUpsert(node("node:b")),
    ]))?;
    store.commit_batch(rustyred_thg_core::GraphMutationBatch::new([
        rustyred_thg_core::GraphMutation::EdgeUpsert(edge(
            "edge:first",
            "node:a",
            "node:b",
            LINK_TYPE,
        )),
        rustyred_thg_core::GraphMutation::EdgeUpsert(edge(
            "edge:second",
            "node:a",
            "node:b",
            LINK_TYPE,
        )),
    ]))?;
    store.delete_edge("edge:first")?;
    drop(store);
    let log = PhysicalLog::open(directory.path(), deployment_id())?;
    let query = json!({
        "kind": "reachable",
        "edge_type": LINK_TYPE,
        "sources": ["node:a"]
    });
    let mut algebra = FoldRuntime::for_log(AlgebraFold::default(), &log)?;
    algebra.apply_range(&log, LogPosition(2))?;
    algebra.upquery(&log, &query, LogPosition(2))?;
    algebra.apply_range(&log, LogPosition(3))?;
    let answer = decode_answer(algebra.query(&query, LogPosition(3))?)?;
    assert_eq!(answer.reachable, vec!["node:a", "node:b"]);
    assert_eq!(algebra.inner().residency()?.matrix_entries, 1);
    Ok(())
}

#[test]
fn unsupported_log_records_are_quarantined_without_native_state() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let mut log = PhysicalLog::open(directory.path(), deployment_id())?;
    log.append(
        LogIntent::new(
            IntentKind::new("fixture", "unrelated_event", 1)?,
            json!({"value": 1}),
        )?,
        WriteSet::new([AccessPath::Entity {
            entity: "fixture:one".to_string(),
        }])?,
        OpDigest::from_bytes("algebra-unsupported-record"),
    )?;
    let mut algebra = FoldRuntime::for_log(AlgebraFold::default(), &log)?;
    let report = algebra.apply_all(&log)?;
    assert_eq!(report.applied, 0);
    assert_eq!(report.quarantined, 1);
    assert_eq!(algebra.inner().residency()?.matrix_entries, 0);
    assert_eq!(algebra.inner().stats().interval_batches, 0);
    Ok(())
}

fn build_fixture(update_count: usize) -> Result<Fixture, Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let mut store = FoldGraphStore::open(directory.path(), deployment_id())?;
    store.commit_batch(rustyred_thg_core::GraphMutationBatch::new((0..16).map(
        |index| rustyred_thg_core::GraphMutation::NodeUpsert(node(&format!("node:{index}"))),
    )))?;
    let mut snapshots = vec![store.graph_snapshot()?];
    let mut random = DeterministicRandom::new(SEED);
    let mut active = BTreeSet::new();

    for step in 0..update_count {
        let slot = random.usize(32);
        let edge_id = format!("edge:{slot}");
        if active.contains(&slot) && random.next() % 4 == 0 {
            store.delete_edge(&edge_id)?;
            active.remove(&slot);
        } else {
            let from = random.usize(16);
            let mut to = random.usize(16);
            if to == from {
                to = (to + 1) % 16;
            }
            let edge_type = if step == 17 || random.next() % 3 == 0 {
                LINK_TYPE
            } else {
                "OTHER"
            };
            store.upsert_edge(edge(
                &edge_id,
                &format!("node:{from}"),
                &format!("node:{to}"),
                edge_type,
            ))?;
            active.insert(slot);
        }
        snapshots.push(store.graph_snapshot()?);
    }

    drop(store);
    let log = PhysicalLog::open(directory.path(), deployment_id())?;
    assert_eq!(snapshots.len() as u64, log.latest_position().0);
    Ok(Fixture {
        _directory: directory,
        log,
        snapshots,
    })
}

fn oracle_at(fixture: &Fixture, position: LogPosition, source: &str) -> Vec<String> {
    let snapshot = &fixture.snapshots[(position.0 - 1) as usize];
    let live = snapshot
        .nodes
        .iter()
        .filter(|node| !node.tombstone)
        .map(|node| node.id.clone())
        .collect::<BTreeSet<_>>();
    let mut outgoing = BTreeMap::<String, BTreeSet<String>>::new();
    for edge in snapshot
        .edges
        .iter()
        .filter(|edge| !edge.tombstone && edge.edge_type == LINK_TYPE)
    {
        if live.contains(&edge.from_id) && live.contains(&edge.to_id) {
            outgoing
                .entry(edge.from_id.clone())
                .or_default()
                .insert(edge.to_id.clone());
        }
    }
    let mut reached = BTreeSet::from([source.to_string()]);
    let mut pending = VecDeque::from([source.to_string()]);
    while let Some(node) = pending.pop_front() {
        for successor in outgoing.get(&node).into_iter().flatten() {
            if reached.insert(successor.clone()) {
                pending.push_back(successor.clone());
            }
        }
    }
    reached.into_iter().collect()
}

fn decode_answer(value: serde_json::Value) -> Result<AlgebraReachability, serde_json::Error> {
    serde_json::from_value(value)
}

fn reachable_query(source: &str) -> serde_json::Value {
    json!({
        "kind": "reachable",
        "edge_type": LINK_TYPE,
        "sources": [source]
    })
}

fn node(id: &str) -> NodeRecord {
    NodeRecord::new(id, ["AlgebraFixture"], json!({"fixture": true}))
}

fn edge(id: &str, from: &str, to: &str, edge_type: &str) -> EdgeRecord {
    EdgeRecord::new(id, from, edge_type, to, json!({"fixture": true}))
}

fn deployment_id() -> theorem_oplog::DeploymentId {
    theorem_oplog::DeploymentId::new("test:algebra-fold-differential").unwrap()
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
