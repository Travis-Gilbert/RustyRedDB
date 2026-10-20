use rustyred_thg_core::{
    export_records_jsonl, ContextView, ContextViewType, FreshnessStatus, HydrationHandle,
    LabeledTrainingRun, MapArtifact, MapArtifactType, MapSection, ReceiptScope, RedactionStatus,
    TrainingExportKind, TrainingExportRecord, TrainingLabel, TrainingLabelFamily, TrainingTaskType,
};

fn scope() -> ReceiptScope {
    ReceiptScope::from([
        ("tenant".to_string(), "Travis-Gilbert".to_string()),
        ("repo".to_string(), "Travis-Gilbert/Theorem".to_string()),
    ])
}

#[test]
fn context_view_rejects_summary_only_memory() {
    let mut view = ContextView::new(
        "view:repo-onboarding:1",
        ContextViewType::RepoOnboarding,
        scope(),
        42,
        "Architecture, tests, and sharp edges.",
    );

    let error = view.validate_not_summary_only().unwrap_err();
    assert_eq!(error.code, "summary_only_context_view");

    view.included_atom_ids.push("atom:graph-store".to_string());
    view.excluded_atom_ids
        .push("atom:stale-build-note".to_string());
    view.add_hydration_handle(HydrationHandle::new(
        "atom:graph-store",
        "ContextAtom",
        42,
        "graph://ContextAtom/atom:graph-store",
    ));
    view.validate_not_summary_only().unwrap();

    view.mark_stale();
    assert_eq!(view.freshness_status, FreshnessStatus::Stale);
    assert_eq!(view.version, 2);
}

#[test]
fn map_sections_record_reuse_as_positive_and_negative_labels() {
    let mut map = MapArtifact::new(
        "map:codebase:theorem",
        MapArtifactType::Codebase,
        scope(),
        42,
    );
    map.add_section(MapSection::new(
        "section:rustyred-core",
        "RustyRed core",
        "GraphStore, planner, indexes, and durable RedCore storage.",
    ));

    map.record_section_usage(
        "section:rustyred-core",
        true,
        "label:used-in-successful-run".to_string(),
    );
    map.record_section_usage(
        "section:rustyred-core",
        false,
        "label:ignored-in-review".to_string(),
    );

    let section = map.section("section:rustyred-core").unwrap();
    assert_eq!(section.usage_count, 2);
    assert_eq!(
        section.positive_label_ids,
        vec!["label:used-in-successful-run"]
    );
    assert_eq!(section.negative_label_ids, vec!["label:ignored-in-review"]);
}

#[test]
fn labeled_training_run_requires_positive_and_negative_context_evidence_for_export() {
    let mut run = LabeledTrainingRun::new(
        "training:1",
        "run:1",
        "artifact:1",
        TrainingTaskType::Review,
        "Codex",
        scope(),
        42,
    );
    run.candidate_atom_ids = vec!["atom:a".to_string(), "atom:b".to_string()];
    run.included_atom_ids = vec!["atom:a".to_string()];
    run.cited_atom_ids = vec!["atom:a".to_string()];
    assert!(!run.is_exportable(), "negative labels are required");

    run.dismissed_atom_ids = vec!["atom:b".to_string()];
    run.add_label(TrainingLabel::new(
        TrainingLabelFamily::ContextRanking,
        "dismissed",
        "atom:b",
    ));
    assert!(run.is_exportable());

    let record = TrainingExportRecord::from_labeled_run(
        TrainingExportKind::ContextRank,
        &run,
        RedactionStatus::Safe,
    )
    .unwrap();
    let jsonl = export_records_jsonl(&[record]).unwrap();

    assert!(jsonl.contains("\"graph_version\":42"));
    assert!(jsonl.contains("\"redaction_status\":\"safe\""));
    assert!(jsonl.ends_with('\n'));
}
