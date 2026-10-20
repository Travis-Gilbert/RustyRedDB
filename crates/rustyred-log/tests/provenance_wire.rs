use theorem_oplog::{
    BridgeTransport, CommitAuthority, EntryOrigin, EntryProvenance, EvidenceGrade,
    VerificationOutcome, VerificationState,
};

#[test]
fn pre_move_transport_tokens_round_trip_byte_for_byte() {
    for golden in [r#""in_process""#, r#""py_o3""#, r#""service""#] {
        let decoded: BridgeTransport = serde_json::from_str(golden).unwrap();
        assert_eq!(serde_json::to_string(&decoded).unwrap(), golden);
    }
}

#[test]
fn pre_move_verification_tokens_round_trip_byte_for_byte() {
    for golden in [
        r#""unverified""#,
        r#""feasibility_verified""#,
        r#""solver_attested""#,
        r#""grounding_verified""#,
        r#""source_attested""#,
        r#""failed""#,
    ] {
        let decoded: VerificationState = serde_json::from_str(golden).unwrap();
        assert_eq!(serde_json::to_string(&decoded).unwrap(), golden);
    }
}

#[test]
fn legacy_provenance_default_is_unknown_not_fabricated() {
    let decoded: EntryProvenance = serde_json::from_str("{}").unwrap();
    assert_eq!(decoded.origin, EntryOrigin::LegacyUnknown);
    assert_eq!(decoded.authority, CommitAuthority::LegacyUnknown);
    assert_eq!(decoded.transport, None);
    assert_eq!(decoded.verification, VerificationState::Unverified);
    assert_eq!(decoded.writer, None);
    assert!(decoded.is_explicit_legacy());
}

#[test]
fn outcome_grade_and_display_rank_remain_distinct_contracts() {
    assert_eq!(
        VerificationState::SourceAttested.outcome(),
        VerificationOutcome::Supported
    );
    assert_eq!(
        VerificationState::SourceAttested.evidence_grade(),
        Some(EvidenceGrade::Attested)
    );
    assert_eq!(
        VerificationState::Failed.outcome(),
        VerificationOutcome::Refuted
    );
    assert_eq!(VerificationState::Failed.evidence_grade(), None);

    let ranks = [
        VerificationState::Failed,
        VerificationState::Unverified,
        VerificationState::SourceAttested,
        VerificationState::SolverAttested,
        VerificationState::FeasibilityVerified,
        VerificationState::GroundingVerified,
    ]
    .map(VerificationState::display_rank);
    assert_eq!(ranks, [0, 1, 2, 3, 4, 5]);
}

#[test]
fn origin_authority_matrix_refuses_query_append() {
    assert!(CommitAuthority::Mutation.permits(EntryOrigin::Mutation));
    assert!(CommitAuthority::Mutation.permits(EntryOrigin::Action));
    assert!(!CommitAuthority::Mutation.permits(EntryOrigin::Query));
}
