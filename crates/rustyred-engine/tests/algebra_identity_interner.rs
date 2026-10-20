//! Compatibility oracle for SPEC-THEOREM-MEMTENSOR-IDENTITY-INTERNER-1.0
//! Option A: node identity is an in-memory symbol, and every durability
//! boundary stays string-keyed.
//!
//! The Algebra checkpoint is the boundary the F6 write-up did not name.
//! `AlgebraFold::checkpoint` serializes `nodes_by_index` and `live_nodes` into
//! the `FoldCheckpoint.state` bytes that `checkpoint_hash` covers, and
//! `restore` reads them back under an unchanged `ALGEBRA_FOLD_VERSION`. If
//! symbols ever reached that document, every stored checkpoint would become
//! unreadable while every content-id assertion still passed. These tests are
//! what stops that.

use rustyred_thg_core::{
    AlgebraFold, DeploymentId, FoldGraphStore, FoldRuntime, GraphWrite, NodeIndex, NodeRecord,
    RestoreDisposition,
};
use serde_json::{json, Value};
use std::error::Error;
use theorem_oplog::PhysicalLog;

const IDS: [&str; 4] = ["memory:zulu", "memory:alpha", "memory:mike", "memory:bravo"];

/// Admit four rows in an order that is deliberately not lexicographic, so a
/// symbol-ordered or hash-ordered document would be visibly different from the
/// string-ordered one this asserts.
fn seeded_log(label: &str) -> Result<(tempfile::TempDir, PhysicalLog), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let deployment = DeploymentId::new(label)?;
    let mut store = FoldGraphStore::open(directory.path(), deployment.clone())?;
    for id in IDS {
        GraphWrite::upsert_node(&mut store, NodeRecord::new(id, ["Memory"], json!({})))?;
    }
    // One retracted row: it leaves `nodes_by_index` but not `live_nodes`, so
    // the two arrays cannot be confused for each other.
    GraphWrite::upsert_node(
        &mut store,
        NodeRecord::new("memory:retracted", ["Memory"], json!({})),
    )?;
    store.delete_node("memory:retracted")?;
    drop(store);
    let log = PhysicalLog::open(directory.path(), deployment)?;
    Ok((directory, log))
}

fn checkpoint_state(log: &PhysicalLog) -> Result<Value, Box<dyn Error>> {
    let mut runtime = FoldRuntime::for_log(AlgebraFold::default(), log)?;
    runtime.apply_all(log)?;
    Ok(serde_json::from_slice(&runtime.checkpoint(log)?.state)?)
}

#[test]
fn checkpoint_identity_stays_string_keyed_and_index_ordered() -> Result<(), Box<dyn Error>> {
    let (_directory, log) = seeded_log("algebra-interner-checkpoint")?;
    let state = checkpoint_state(&log)?;

    let nodes = state["nodes_by_index"]
        .as_array()
        .expect("nodes_by_index is an array");
    let nodes = nodes
        .iter()
        .map(|value| value.as_str().expect("node identity is a string"))
        .collect::<Vec<_>>();
    assert_eq!(
        nodes,
        [
            "memory:zulu",
            "memory:alpha",
            "memory:mike",
            "memory:bravo",
            "memory:retracted"
        ],
        "the checkpoint carries canonical strings in row order, not symbols"
    );

    let live = state["live_nodes"]
        .as_array()
        .expect("live_nodes is an array")
        .iter()
        .map(|value| value.as_str().expect("live identity is a string"))
        .collect::<Vec<_>>();
    assert_eq!(
        live,
        ["memory:alpha", "memory:bravo", "memory:mike", "memory:zulu"],
        "live_nodes stays a sorted string set: the in-memory set is symbol-keyed \
         and unordered, so serializing it unsorted would silently move checkpoint_hash"
    );
    Ok(())
}

#[test]
fn a_string_keyed_checkpoint_restores_to_the_same_identities() -> Result<(), Box<dyn Error>> {
    let (_directory, log) = seeded_log("algebra-interner-restore")?;
    let mut source = FoldRuntime::for_log(AlgebraFold::default(), &log)?;
    source.apply_all(&log)?;
    let checkpoint = source.checkpoint(&log)?;

    let mut target = FoldRuntime::for_log(AlgebraFold::default(), &log)?;
    let RestoreDisposition::Restored { .. } = target.restore_or_rebuild(&checkpoint, &log)? else {
        panic!("a same-version Algebra checkpoint must restore, not rebuild");
    };

    // Identity survives the string -> symbol -> string round trip, and the
    // reverse index the restore rebuilds still points at the right rows.
    let restored = target.inner();
    for (row, id) in IDS.iter().enumerate() {
        assert_eq!(
            restored.node_index(id),
            Some(NodeIndex(row as u64)),
            "restored fold lost the row index of {id}"
        );
        assert_eq!(restored.node_id(NodeIndex(row as u64)), Some(*id));
    }
    assert!(
        restored.node_index("memory:absent").is_none(),
        "an id the fold never admitted resolves to nothing"
    );

    // The whole point of keeping the document string-keyed: re-checkpointing a
    // restored fold reproduces the same bytes, so `checkpoint_hash` is stable
    // across a restart.
    assert_eq!(
        target.checkpoint(&log)?.state,
        checkpoint.state,
        "checkpoint bytes must survive a restore round trip"
    );
    Ok(())
}
