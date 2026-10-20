use rustyred_thg_core::{
    FreshnessStatus, HydrationHandle, MapArtifact, MapArtifactType, MapSection, ReceiptScope,
};

fn scope() -> ReceiptScope {
    ReceiptScope::from([
        ("tenant".to_string(), "Travis-Gilbert".to_string()),
        ("project".to_string(), "Theorem".to_string()),
    ])
}

fn section(id: &str, title: &str, summary: &str, graph_version: u64) -> MapSection {
    let mut section = MapSection::new(id, title, summary);
    section.hydration_handles.push(HydrationHandle::new(
        format!("atom:{id}"),
        "ContextAtom",
        graph_version,
        format!("graph://ContextAtom/atom:{id}"),
    ));
    section.source_atom_ids.push(format!("atom:{id}"));
    section
}

#[test]
fn map_diff_and_refresh_preserve_section_identity_and_mark_stale_sources() {
    let mut current = MapArtifact::new("map:project", MapArtifactType::Project, scope(), 10);
    current.add_section(section("overview", "Overview", "old summary", 9));
    current.add_section(section("removed", "Removed", "obsolete", 10));

    assert!(current.mark_stale_if_graph_version_behind(11));
    assert_eq!(current.freshness_status, FreshnessStatus::NeedsRebuild);
    assert_eq!(
        current.section("overview").unwrap().freshness_status,
        FreshnessStatus::Stale
    );

    let mut regenerated = MapArtifact::new("map:project", MapArtifactType::Project, scope(), 11);
    regenerated.add_section(section("overview", "Overview", "new summary", 11));
    regenerated.add_section(section("added", "Added", "new area", 11));

    let diff = current.refresh_from(regenerated);

    assert_eq!(diff.from_version, 1);
    assert_eq!(diff.to_version, 2);
    assert_eq!(diff.added_section_ids, vec!["added"]);
    assert_eq!(diff.removed_section_ids, vec!["removed"]);
    assert_eq!(diff.changed_section_ids, vec!["overview"]);
    assert_eq!(diff.stale_section_ids, vec!["overview", "removed"]);
    assert_eq!(current.version, 2);
    assert_eq!(current.graph_version, 11);
    assert_eq!(current.freshness_status, FreshnessStatus::Fresh);
    assert_eq!(
        current.section("overview").unwrap().freshness_status,
        FreshnessStatus::Fresh
    );
}
