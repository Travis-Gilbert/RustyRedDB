use super::*;
use crate::file_payload::{
    install_file_manifest_reference, read_file_payload, stage_file_payload, FileManifestRef,
    FILE_CONTENT_MANIFEST_PROPERTY,
};
use crate::graph_history::GraphHistoryEntry;
use crate::graph_store::{RedCoreDurability, RedCoreOptions};
use std::path::Path;

const SOURCE: &str = "personal-store";

fn options() -> RedCoreOptions {
    RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 0,
        strict_acid: true,
    }
}

fn open(dir: &Path) -> RedCoreGraphStore {
    RedCoreGraphStore::open(dir, options()).expect("open store")
}

fn metadata(n: u64) -> GraphCommitMetadata {
    GraphCommitMetadata {
        author: format!("verified-user:{n}"),
        message: format!("authored edit {n}"),
        timestamp_unix_ms: 1_700_000_000_000 + u128::from(n),
    }
}

fn commit(
    store: &mut RedCoreGraphStore,
    n: u64,
    mutations: impl IntoIterator<Item = GraphMutation>,
) -> crate::graph_history::GraphCommitReceipt {
    store
        .commit_batch_with_metadata(GraphMutationBatch::new(mutations), metadata(n))
        .expect("commit")
}

fn article(id: &str, body: &str) -> GraphMutation {
    GraphMutation::NodeUpsert(NodeRecord::new(id, ["Memory"], json!({ "body": body })))
}

fn link(id: &str, from: &str, to: &str) -> GraphMutation {
    GraphMutation::EdgeUpsert(EdgeRecord::new(id, from, "LINKS", to, json!({})))
}

fn import_options() -> GraphImportOptions {
    GraphImportOptions {
        source_id: SOURCE.to_owned(),
        ..GraphImportOptions::default()
    }
}

fn body(store: &RedCoreGraphStore, id: &str) -> Option<String> {
    store
        .pin_graph_history()
        .unwrap()
        .head_revision()
        .unwrap()
        .get_node(id)
        .unwrap()
        .filter(|node| !node.tombstone)
        .map(|node| node.properties["body"].as_str().unwrap().to_owned())
}

fn all_entries(store: &RedCoreGraphStore) -> Vec<GraphHistoryEntry> {
    let history = store.pin_graph_history().unwrap();
    let mut entries = Vec::new();
    let mut after = 0;
    loop {
        let page = history.entries_after(after, MAX_HISTORY_PAGE).unwrap();
        let Some(last) = page.last() else {
            return entries;
        };
        after = last.roots.transaction;
        entries.extend(page);
    }
}

fn head_transaction(store: &RedCoreGraphStore) -> u64 {
    store.pin_graph_history().unwrap().head_roots().transaction
}

#[test]
fn complete_history_replays_every_commit_with_its_authorship() {
    let (source_dir, target_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let mut source = open(source_dir.path());
    commit(
        &mut source,
        1,
        [
            article("article:a", "body 1"),
            article("article:b", "other"),
        ],
    );
    commit(&mut source, 2, [link("link:ab", "article:a", "article:b")]);
    commit(&mut source, 3, [article("article:a", "body 2")]);
    commit(
        &mut source,
        4,
        [GraphMutation::EdgeDelete("link:ab".to_owned())],
    );
    let mut target = open(target_dir.path());
    commit(&mut target, 100, [article("target:own", "kept")]);
    let target_before = head_transaction(&target);
    let source_root = source.unified_chunk_store().read_root().unwrap();
    let source_head = head_transaction(&source);

    let plan = plan_graph_import(&source, &target, &import_options()).unwrap();
    assert_eq!(plan.start, GraphImportStart::Complete);
    assert_eq!(plan.commits_to_replay, 4);
    assert_eq!(plan.touched_keys, 3);
    // The source's control events for creating, updating and deleting.
    assert_eq!(plan.derived_keys_skipped, 3);
    assert_eq!(plan.conflict_count, 0);
    assert!(plan.is_admissible());

    let (_, report) = apply_graph_import(&source, &mut target, &import_options()).unwrap();
    assert_eq!(report.commits_written, 4);
    assert_eq!(report.records_deleted, 1);
    assert_eq!(report.last_source_transaction, source_head);

    // Each imported commit carries the source's authorship, in source order,
    // after the target's own history.
    let imported = all_entries(&target)
        .into_iter()
        .filter(|entry| entry.roots.transaction > target_before)
        .map(|entry| {
            (
                entry.commit.author,
                entry.commit.message,
                entry.commit.timestamp_unix_ms,
            )
        })
        .collect::<Vec<_>>();
    let expected = (1..=4)
        .map(|n| {
            let meta = metadata(n);
            (meta.author, meta.message, meta.timestamp_unix_ms)
        })
        .collect::<Vec<_>>();
    assert_eq!(imported, expected);

    assert_eq!(body(&target, "article:a").as_deref(), Some("body 2"));
    assert_eq!(body(&target, "target:own").as_deref(), Some("kept"));
    let head = target.pin_graph_history().unwrap().head_revision().unwrap();
    assert!(head.get_edge("link:ab").unwrap().unwrap().tombstone);
    // A revision in the middle of the import still shows the edge.
    let middle = all_entries(&target)
        .into_iter()
        .find(|entry| entry.commit.message == "authored edit 2")
        .unwrap();
    assert!(target
        .pin_graph_history()
        .unwrap()
        .revision(middle.commit.commit_hash)
        .unwrap()
        .get_edge("link:ab")
        .unwrap()
        .is_some_and(|edge| !edge.tombstone));

    // Both stores numbered a control event `controlevent:1:0:Memory.created`.
    // The target's own, describing its own record, is untouched.
    let own_event = head
        .get_node("controlevent:1:0:Memory.created")
        .unwrap()
        .unwrap();
    assert_eq!(
        own_event.properties["affected_node_ids"],
        json!(["target:own"])
    );

    let verification = verify_graph_import(&source, &target, SOURCE).unwrap();
    assert!(verification.is_conserved(), "{verification:?}");
    assert_eq!(verification.source_records, 2);
    assert_eq!(verification.source_derived_records, 3);
    assert_eq!(verification.target_only_records, 1);
    assert_eq!(
        verification.by_label["Memory"],
        GraphImportLabelCount {
            source: 2,
            matched: 2
        }
    );
    assert!(target.verify().unwrap().ok);

    // The source was only read.
    assert_eq!(
        source.unified_chunk_store().read_root().unwrap(),
        source_root
    );
    assert_eq!(head_transaction(&source), source_head);
}

#[test]
fn a_rerun_writes_nothing_and_a_catch_up_survives_restart() {
    let (source_dir, target_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let mut source = open(source_dir.path());
    commit(&mut source, 1, [article("article:a", "body 1")]);
    {
        let mut target = open(target_dir.path());
        apply_graph_import(&source, &mut target, &import_options()).unwrap();
        let before = head_transaction(&target);
        let (plan, report) = apply_graph_import(&source, &mut target, &import_options()).unwrap();
        assert_eq!(plan.commits_to_replay, 0);
        assert_eq!(report.commits_written, 0);
        assert_eq!(head_transaction(&target), before);
    }

    commit(&mut source, 2, [article("article:b", "new since")]);
    let mut target = open(target_dir.path());
    let plan = plan_graph_import(&source, &target, &import_options()).unwrap();
    assert!(matches!(plan.start, GraphImportStart::Resume { .. }));
    assert_eq!(plan.commits_to_replay, 1);
    let (_, report) = apply_graph_import(&source, &mut target, &import_options()).unwrap();
    assert_eq!(report.commits_written, 1);
    assert_eq!(body(&target, "article:b").as_deref(), Some("new since"));
    assert!(verify_graph_import(&source, &target, SOURCE)
        .unwrap()
        .is_conserved());
}

#[test]
fn an_independent_target_edit_refuses_the_whole_run_until_accepted() {
    let (source_dir, target_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let mut source = open(source_dir.path());
    commit(
        &mut source,
        1,
        [
            article("article:a", "source version"),
            article("article:same", "identical"),
        ],
    );
    let mut target = open(target_dir.path());
    let own = commit(
        &mut target,
        100,
        [
            article("article:a", "target version"),
            article("article:same", "identical"),
        ],
    );
    let before = head_transaction(&target);

    let plan = plan_graph_import(&source, &target, &import_options()).unwrap();
    // The identical record converges; only the differing one conflicts.
    assert_eq!(plan.converged_keys, 1);
    assert_eq!(plan.conflict_count, 1);
    let conflict = &plan.conflicts[0];
    assert_eq!(conflict.key, "node/article:a");
    assert_eq!(conflict.base_content_address, None);
    let target_address = conflict.target_content_address.clone().unwrap();

    let refusal = apply_graph_import(&source, &mut target, &import_options()).unwrap_err();
    assert_eq!(refusal.code, "graph_import_refused");
    assert_eq!(head_transaction(&target), before);
    assert_eq!(
        body(&target, "article:a").as_deref(),
        Some("target version")
    );

    // A stale acceptance (wrong address) still refuses.
    let mut stale = import_options();
    stale
        .accepted_overwrites
        .insert("node/article:a".to_owned(), "sha256:stale".to_owned());
    assert!(apply_graph_import(&source, &mut target, &stale).is_err());

    let mut accepted = import_options();
    accepted
        .accepted_overwrites
        .insert("node/article:a".to_owned(), target_address);
    let (plan, _) = apply_graph_import(&source, &mut target, &accepted).unwrap();
    assert_eq!(plan.accepted_overwrites, vec!["node/article:a".to_owned()]);
    assert_eq!(
        body(&target, "article:a").as_deref(),
        Some("source version")
    );
    // The overwritten value is still in the target's own history.
    let prior = target
        .pin_graph_history()
        .unwrap()
        .revision(own.commit.commit_hash)
        .unwrap()
        .get_node("article:a")
        .unwrap()
        .unwrap();
    assert_eq!(prior.properties["body"], "target version");
}

#[test]
fn a_target_edit_after_import_blocks_only_the_catch_up_that_touches_it() {
    let (source_dir, target_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let mut source = open(source_dir.path());
    commit(
        &mut source,
        1,
        [article("article:a", "one"), article("article:b", "one")],
    );
    let mut target = open(target_dir.path());
    apply_graph_import(&source, &mut target, &import_options()).unwrap();
    commit(
        &mut target,
        100,
        [article("article:a", "edited in the target")],
    );

    commit(&mut source, 2, [article("article:b", "two")]);
    let plan = plan_graph_import(&source, &target, &import_options()).unwrap();
    assert_eq!(plan.conflict_count, 0, "an untouched key does not conflict");

    commit(&mut source, 3, [article("article:a", "two")]);
    let plan = plan_graph_import(&source, &target, &import_options()).unwrap();
    assert_eq!(plan.conflict_count, 1);
    assert_eq!(plan.conflicts[0].key, "node/article:a");
    assert!(plan.conflicts[0].base_content_address.is_some());
}

#[test]
fn file_bodies_and_vectors_land_before_the_records_that_reference_them() {
    let (source_dir, target_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let mut source = open(source_dir.path());
    source
        .designate_vector_property("Memory", "embedding", 3)
        .unwrap();
    let file_body = (0..200_000u32)
        .map(|n| (n.wrapping_mul(2_654_435_761) >> 13) as u8)
        .collect::<Vec<_>>();
    let chunk_store = source.unified_chunk_store();
    let staged = stage_file_payload(
        chunk_store.as_ref(),
        &format!("graph-node:file:notes/a.txt:property:{FILE_CONTENT_MANIFEST_PROPERTY}"),
        &file_body,
    )
    .unwrap();
    let reference: FileManifestRef = staged.reference().clone();
    let body_chunks = staged.manifest().chunks.len();
    assert!(body_chunks > 1, "the fixture body must span several chunks");
    let mut properties = json!({ "path": "notes/a.txt" });
    install_file_manifest_reference(properties.as_object_mut().unwrap(), &reference, 0).unwrap();
    commit(
        &mut source,
        1,
        [
            GraphMutation::NodeUpsert(NodeRecord::new("file:notes/a.txt", ["File"], properties)),
            GraphMutation::NodeUpsert(NodeRecord::new(
                "memory:a",
                ["Memory"],
                json!({ "embedding": [0.25, 0.5, 0.75] }),
            )),
        ],
    );
    staged.finish_after_publication();

    let mut target = open(target_dir.path());
    let (_, report) = apply_graph_import(&source, &mut target, &import_options()).unwrap();
    // Manifest, every body chunk, and the vector payload.
    assert_eq!(report.chunks_copied, body_chunks + 2);
    assert!(report
        .designations_applied
        .contains(&"vector Memory.embedding (3)".to_owned()));
    assert!(target.vector_designations().iter().any(|designation| {
        designation.label == "Memory" && designation.property == "embedding"
    }));
    assert_eq!(
        read_file_payload(target.unified_chunk_store().as_chunk_store(), &reference).unwrap(),
        file_body
    );
    let verification = verify_graph_import(&source, &target, SOURCE).unwrap();
    assert!(verification.is_conserved(), "{verification:?}");
    assert_eq!(verification.payload_refs_checked, body_chunks + 2);
    assert!(target.verify().unwrap().ok);

    // The copied bodies survive a reopen of the target.
    drop(target);
    let target = open(target_dir.path());
    assert_eq!(
        read_file_payload(target.unified_chunk_store().as_chunk_store(), &reference).unwrap(),
        file_body
    );
}

#[test]
fn a_source_without_its_earliest_history_starts_from_one_baseline() {
    let (source_dir, target_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let legacy_transaction;
    {
        let mut source = open(source_dir.path());
        let first = commit(
            &mut source,
            1,
            [
                article("article:a", "body 1"),
                article("article:b", "other"),
            ],
        );
        legacy_transaction = first.transaction.txn_id;
        source.publish_v4_state_layout_for_tests().unwrap();
    }
    let mut source = open(source_dir.path());
    let target = open(target_dir.path());
    // Nothing retained yet: the baseline is the head itself.
    let plan = plan_graph_import(&source, &target, &import_options()).unwrap();
    assert_eq!(plan.coverage, GraphHistoryCoverage::Unavailable);
    assert_eq!(
        plan.start,
        GraphImportStart::Baseline {
            transaction: legacy_transaction,
            commit: None
        }
    );
    assert_eq!(plan.commits_to_replay, 0);
    drop(target);

    commit(&mut source, 2, [article("article:a", "body 2")]);
    let mut target = open(target_dir.path());
    let plan = plan_graph_import(&source, &target, &import_options()).unwrap();
    assert!(matches!(plan.coverage, GraphHistoryCoverage::Since { .. }));
    assert_eq!(plan.baseline_records, 2);
    assert_eq!(plan.commits_to_replay, 1);

    let (_, report) = apply_graph_import(&source, &mut target, &import_options()).unwrap();
    assert!(report.baseline_written);
    assert_eq!(report.commits_written, 2);
    let entries = all_entries(&target);
    assert_eq!(entries[0].commit.author, GRAPH_IMPORT_AUTHOR);
    assert_eq!(entries[1].commit.message, "authored edit 2");
    let baseline = target
        .pin_graph_history()
        .unwrap()
        .revision(entries[0].commit.commit_hash)
        .unwrap();
    assert_eq!(
        baseline.get_node("article:a").unwrap().unwrap().properties["body"],
        "body 1"
    );
    assert_eq!(body(&target, "article:a").as_deref(), Some("body 2"));
    assert!(verify_graph_import(&source, &target, SOURCE)
        .unwrap()
        .is_conserved());
}

#[test]
fn records_naming_an_unexpected_tenant_refuse_the_import() {
    let (source_dir, target_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let mut source = open(source_dir.path());
    commit(
        &mut source,
        1,
        [
            GraphMutation::NodeUpsert(NodeRecord::new(
                "mine",
                ["Memory"],
                json!({ "tenant": "Travis-Gilbert" }),
            )),
            GraphMutation::NodeUpsert(NodeRecord::new(
                "theirs",
                ["Memory"],
                json!({ "tenant_id": "someone-else" }),
            )),
        ],
    );
    let mut target = open(target_dir.path());

    let plan = plan_graph_import(&source, &target, &import_options()).unwrap();
    assert_eq!(plan.tenant_census["tenant=Travis-Gilbert"], 1);
    assert_eq!(plan.tenant_census["tenant_id=someone-else"], 1);
    assert!(plan.is_admissible(), "no expectation, census only");

    let mut expecting = import_options();
    expecting.expected_tenants = Some(BTreeSet::from(["Travis-Gilbert".to_owned()]));
    let plan = plan_graph_import(&source, &target, &expecting).unwrap();
    assert_eq!(
        plan.foreign_tenants,
        vec!["tenant_id=someone-else".to_owned()]
    );
    let refusal = apply_graph_import(&source, &mut target, &expecting).unwrap_err();
    assert_eq!(refusal.code, "graph_import_refused");
    assert!(target.pin_graph_history().is_err() || head_transaction(&target) == 0);
}

#[test]
fn a_bounded_run_stops_at_its_limit_and_the_next_run_resumes() {
    let (source_dir, target_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let mut source = open(source_dir.path());
    for n in 1..=3 {
        commit(&mut source, n, [article("article:a", &format!("body {n}"))]);
    }
    let mut target = open(target_dir.path());
    let mut bounded = import_options();
    bounded.max_commits = Some(2);

    let (plan, report) = apply_graph_import(&source, &mut target, &bounded).unwrap();
    assert_eq!(plan.commits_to_replay, 2);
    assert_eq!(report.commits_written, 2);
    assert_eq!(body(&target, "article:a").as_deref(), Some("body 2"));
    assert!(verify_graph_import(&source, &target, SOURCE)
        .unwrap()
        .is_conserved());

    let (_, report) = apply_graph_import(&source, &mut target, &bounded).unwrap();
    assert_eq!(report.commits_written, 1);
    assert_eq!(body(&target, "article:a").as_deref(), Some("body 3"));
}

#[test]
fn verification_names_a_record_the_target_lost() {
    let (source_dir, target_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let mut source = open(source_dir.path());
    commit(
        &mut source,
        1,
        [article("article:a", "one"), article("article:b", "one")],
    );
    let mut target = open(target_dir.path());
    apply_graph_import(&source, &mut target, &import_options()).unwrap();
    assert!(verify_graph_import(&source, &target, SOURCE)
        .unwrap()
        .is_conserved());

    commit(
        &mut target,
        100,
        [
            GraphMutation::NodeDelete("article:a".to_owned()),
            article("article:b", "changed"),
        ],
    );
    let verification = verify_graph_import(&source, &target, SOURCE).unwrap();
    assert!(!verification.is_conserved());
    assert_eq!(verification.missing_keys, vec!["node/article:a".to_owned()]);
    assert_eq!(
        verification.mismatched_keys,
        vec!["node/article:b".to_owned()]
    );
}
