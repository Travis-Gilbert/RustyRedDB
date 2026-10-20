use super::graph::*;
use crate::graph_history::GraphCommitMetadata;
use crate::{
    EdgeRecord, GraphMutation, GraphMutationBatch, GraphPayloadRecord, GraphStoreResult,
    NodeRecord, RedCoreDurability, RedCoreGraphStore, RedCoreOptions,
};
use egraph::{CostModel, RulePack, Schedule};
use serde_json::json;

fn options() -> RedCoreOptions {
    RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 0,
        strict_acid: true,
    }
}
fn commit(store: &mut RedCoreGraphStore, batch: GraphMutationBatch) -> GraphStoreResult<()> {
    store
        .commit_batch_with_metadata(
            batch,
            GraphCommitMetadata {
                author: "symbolic-oracle".into(),
                message: "actual symbolic graph adapter".into(),
                timestamp_unix_ms: 1_800_000_000_000,
            },
        )
        .map(|_| ())
}
fn scope(ids: &[&str]) -> GraphScope {
    GraphScope {
        node_ids: ids.iter().map(|id| (*id).into()).collect(),
        edge_ids: Default::default(),
        observed_at_ms: 1_800_000_000_000,
    }
}
fn seed(store: &mut RedCoreGraphStore) -> GraphStoreResult<()> {
    commit(
        store,
        GraphMutationBatch::new([
            GraphMutation::NodeUpsert(NodeRecord::new(
                "claim",
                ["Claim"],
                json!({"claim_text":"admitted claim"}),
            )),
            GraphMutation::NodeUpsert(NodeRecord::new(
                "private",
                ["Claim"],
                json!({"claim_text":"excluded neighbor"}),
            )),
            GraphMutation::EdgeUpsert(EdgeRecord::new(
                "dependency",
                "claim",
                "DEPENDS_ON",
                "private",
                json!({}),
            )),
        ]),
    )
}

#[test]
fn symbolic_fact_pack_reads_only_admitted_pinned_records() -> GraphStoreResult<()> {
    let dir = tempfile::tempdir().unwrap();
    let mut store = RedCoreGraphStore::open(dir.path(), options())?;
    seed(&mut store)?;
    let pin = store.pin_graph_history()?.head_revision()?;
    let admitted = scope(&["claim"]);
    let facts = fact_pack_from_graph(&pin, &admitted)?;
    assert_eq!(facts.len(), 1);
    assert_eq!(facts.relations["claim"][0][0], "claim");
    assert!(!serde_json::to_string(&facts)
        .unwrap()
        .contains("excluded neighbor"));
    assert_eq!(fact_pack_from_graph(&pin, &scope(&[]))?.len(), 0);
    let mut invalid = admitted;
    invalid.edge_ids.insert("dependency".into());
    assert!(fact_pack_from_graph(&pin, &invalid).is_err());
    assert!(fact_pack_from_graph(&pin, &scope(&["missing"])).is_err());
    Ok(())
}

#[test]
fn symbolic_derivation_reopens_with_canonical_edges_and_repeats_without_writes(
) -> GraphStoreResult<()> {
    let dir = tempfile::tempdir().unwrap();
    let engine = datalog::Compiled::new(vec!["unsupported_claim".into()]);
    let admitted = scope(&["claim"]);
    {
        let mut store = RedCoreGraphStore::open(dir.path(), options())?;
        seed(&mut store)?;
        let pin = store.pin_graph_history()?.head_revision()?;
        let mut adapter = SymbolicGraph::derivation(&pin, &admitted, &engine)?;
        let receipt = adapter.derivation_receipt().unwrap().clone();
        assert_eq!(receipt.derived.len(), 1);
        apply_derivation(&mut adapter, &receipt)?;
        let batch = adapter.finish()?;
        assert!(batch.mutations.iter().any(|mutation| matches!(mutation,
            GraphMutation::EdgeUpsert(edge) if edge.edge_type == "CONSUMES")));
        commit(&mut store, batch)?;
    }
    let store = RedCoreGraphStore::open(dir.path(), options())?;
    let pin = store.pin_graph_history()?.head_revision()?;
    assert_eq!(pin.get_node("claim")?.unwrap().labels, vec!["Claim"]);
    let mut adapter = SymbolicGraph::derivation(&pin, &admitted, &engine)?;
    let mut receipt = adapter.derivation_receipt().unwrap().clone();
    receipt.stats.wall_time_ns += 999;
    apply_derivation(&mut adapter, &receipt)?;
    assert!(adapter.finish()?.mutations.is_empty());
    let records = pin.records_page("edge/", None, 512)?;
    let edge_types: std::collections::BTreeSet<_> = records
        .iter()
        .filter_map(|record| match record {
            GraphPayloadRecord::Edge(edge) => Some(edge.edge_type.as_str()),
            _ => None,
        })
        .collect();
    for required in ["CONSUMES", "PRODUCES", "UNDER", "VIA"] {
        assert!(
            edge_types.contains(required),
            "missing persisted {required} edge"
        );
    }
    Ok(())
}

#[test]
fn symbolic_changed_or_incomplete_receipt_never_prepares_a_commit() -> GraphStoreResult<()> {
    let dir = tempfile::tempdir().unwrap();
    let mut store = RedCoreGraphStore::open(dir.path(), options())?;
    seed(&mut store)?;
    let pin = store.pin_graph_history()?.head_revision()?;
    let engine = datalog::Compiled::new(vec!["unsupported_claim".into()]);
    for incomplete in [false, true] {
        let mut adapter = SymbolicGraph::derivation(&pin, &scope(&["claim"]), &engine)?;
        let mut receipt = adapter.derivation_receipt().unwrap().clone();
        if incomplete {
            receipt.stats.complete = false;
        } else {
            receipt.derived[0].reason = "fabricated explanation".into();
        }
        assert!(apply_derivation(&mut adapter, &receipt).is_err());
        assert!(adapter.finish().is_err());
    }
    assert_eq!(store.pin_graph_history()?.head_roots(), &pin.roots);
    Ok(())
}

#[test]
fn symbolic_publication_keeps_existing_atomic_root_guard() -> GraphStoreResult<()> {
    let dir = tempfile::tempdir().unwrap();
    let mut store = RedCoreGraphStore::open(dir.path(), options())?;
    seed(&mut store)?;
    let pin = store.pin_graph_history()?.head_revision()?;
    let mut adapter =
        SymbolicGraph::derivation(&pin, &scope(&["claim"]), &datalog::Compiled::default())?;
    let receipt = adapter.derivation_receipt().unwrap().clone();
    apply_derivation(&mut adapter, &receipt)?;
    let batch = adapter.finish()?;
    commit(
        &mut store,
        GraphMutationBatch::new([GraphMutation::NodeUpsert(NodeRecord::new(
            "concurrent",
            ["Value"],
            json!({"body":"another writer"}),
        ))]),
    )?;
    assert!(commit(&mut store, batch).is_err());
    let current = store.pin_graph_history()?.head_revision()?;
    assert!(current.get_node("concurrent")?.is_some());
    assert_eq!(current.get_node("claim")?.unwrap().labels, vec!["Claim"]);
    Ok(())
}

#[test]
fn symbolic_extraction_uses_stored_expression_and_original_value_input() -> GraphStoreResult<()> {
    let dir = tempfile::tempdir().unwrap();
    let mut store = RedCoreGraphStore::open(dir.path(), options())?;
    commit(
        &mut store,
        GraphMutationBatch::new([GraphMutation::NodeUpsert(NodeRecord::new(
            "expression",
            ["Value"],
            json!({"expression":"(Add (Num 7) (Num 0))"}),
        ))]),
    )?;
    let pin = store.pin_graph_history()?.head_revision()?;
    let pack = RulePack::new(
        "(datatype Math (Num i64) (Add Math Math :cost 10)) (ruleset simplify) (rewrite (Add x (Num 0)) x :ruleset simplify)",
        CostModel::default(),
    );
    let mut adapter = SymbolicGraph::extraction(
        &pin,
        &scope(&["expression"]),
        "expression",
        "expression",
        pack,
        Schedule::FixedPoint {
            rulesets: vec!["simplify".into()],
            max_iterations: 20,
        },
    )?;
    let receipt = adapter.extraction_receipt().unwrap().clone();
    assert_eq!(receipt.term, "(Num 7)");
    let mut wrong = receipt.clone();
    wrong.term = "(Num 9)".into();
    assert!(apply_extraction(&mut adapter, &wrong).is_err());
    apply_extraction(&mut adapter, &receipt)?;
    let batch = adapter.finish()?;
    assert!(batch.mutations.iter().any(|mutation| matches!(mutation,
        GraphMutation::EdgeUpsert(edge) if edge.edge_type == "CONSUMES" && edge.to_id == "expression")));
    commit(&mut store, batch)?;
    drop(pin);
    drop(store);
    let reopened = RedCoreGraphStore::open(dir.path(), options())?;
    let current = reopened.pin_graph_history()?.head_revision()?;
    assert_eq!(
        current.get_node("expression")?.unwrap().properties["expression"],
        "(Add (Num 7) (Num 0))"
    );
    Ok(())
}

#[test]
fn symbolic_runtime_absence_publishes_closed_input_assumption() -> GraphStoreResult<()> {
    let dir = tempfile::tempdir().unwrap();
    let mut store = RedCoreGraphStore::open(dir.path(), options())?;
    seed(&mut store)?;
    let pin = store.pin_graph_history()?.head_revision()?;
    let runtime = datalog::Runtime::new(
        "(relation seed (i64)) (relation blocked (i64)) (relation clear (i64)) (seed 1) (rule ((seed x) (not (blocked x))) ((clear x)) :name \"unblocked\")",
    ).unwrap();
    // Program literals are actual rule-pack premises. An explicitly empty
    // admitted graph scope stays empty even though the store contains nodes.
    let mut adapter = SymbolicGraph::derivation(&pin, &scope(&[]), &runtime)?;
    let receipt = adapter.derivation_receipt().unwrap().clone();
    let absence = receipt
        .provenance
        .iter()
        .flat_map(|proof| &proof.negative_premises)
        .next()
        .unwrap();
    assert_eq!(absence.fact_pack, receipt.fact_pack);
    apply_derivation(&mut adapter, &receipt)?;
    let batch = adapter.finish()?;
    let assumption = batch
        .mutations
        .iter()
        .find_map(|mutation| match mutation {
            GraphMutation::NodeUpsert(node) if node.id.starts_with("symbolic:absence:") => {
                Some(node)
            }
            _ => None,
        })
        .unwrap();
    assert!(assumption.labels.iter().any(|label| label == "Assumption"));
    assert_eq!(
        assumption.properties["evidence"],
        serde_json::to_value(absence).unwrap()
    );
    assert_eq!(
        assumption.properties["scope"],
        format!("closed_fact_pack:{}", receipt.fact_pack)
    );
    assert!(batch.mutations.iter().any(|mutation| matches!(mutation,
        GraphMutation::EdgeUpsert(edge) if edge.edge_type == "UNDER" && edge.to_id == assumption.id)));
    commit(&mut store, batch)?;
    Ok(())
}
