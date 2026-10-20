//! THG-Core: Theorem HotGraph command runtime.
//!
//! This crate has no Django, Python, or network-server dependencies. Both
//! PyO3 in-process bindings and the standalone HTTP server call this same
//! executor.

pub mod abduction;
pub mod access_method;
pub mod adaptive_index;
pub mod analogy;
pub mod analogy_pass;
pub mod analogy_thompson;
pub mod article_fold;
pub mod assert_claim;
pub mod browser_projection;
pub mod browser_storage;
pub mod chunk_store;
pub mod cold_fragments;
pub mod cold_index;
pub mod colt;
pub mod commands;
pub mod context_view;
pub mod corroboration;
pub mod crdt;
pub mod data_api;
pub mod data_registry;
pub mod derivation;
pub mod derivation_bridge;
pub mod doc_tree;
pub mod dynamic_ppr;
pub mod epistemic;
pub mod errors;
pub mod events;
pub mod executor;
pub mod feature_dsl;
pub mod federated_blocks;
pub mod file_lock;
pub mod file_payload;
pub mod fold;
pub mod fold_quarantine;
pub mod fold_storage;
pub mod fulltext;
#[cfg(feature = "tantivy")]
pub mod fulltext_tantivy;
pub mod graph;
pub mod graph_csr;
pub mod graph_history;
pub mod graph_import;
pub mod graph_store;
/// D2-D6: GraphBLAS sparse-matrix compute (typed adjacency, semiring traversal,
/// LAGraph algorithms, CFL-reachability). Requires the `graphblas` feature.
#[cfg(feature = "graphblas")]
pub mod graphblas_adjacency;
/// D5: CFL-reachability dataflow (points-to / taint) over the typed adjacency.
#[cfg(feature = "graphblas")]
pub mod graphblas_cfl;
/// D6: graph-analytic + reachability operators as first-class hybrid plan nodes.
#[cfg(feature = "graphblas")]
pub mod graphblas_plan;
pub mod hash;
pub mod history_object_store;
pub mod hooks;
pub mod identity_index;
pub mod incremental;
pub mod incremental_topology;
pub mod index;
pub mod index_advisor;
pub mod index_manifest;
pub mod index_proposal;
pub mod index_registry;
pub mod instant_kg;
pub mod intern;
pub mod journal;
pub mod judgment;
pub mod labeled_training_run;
pub mod layer;
pub mod map_artifact;
pub mod memory_admission;
pub mod memory_document;
pub mod merge_registry;
pub mod motif;
pub mod mutation_log;
pub mod narrative;
pub mod node_state;
pub mod object_store;
pub mod ordered;
pub mod partial_fold;
pub mod planner;
pub mod plugin;
pub mod ppr_cache;
pub mod provenance_semiring;
pub mod query_receipt;
pub mod ranking;
pub mod reachability;
pub mod read_model_index;
mod recovery_profile;
pub mod refinery;
pub mod relational;
mod resolved_row;
pub mod revisable;
pub mod rr_file;
pub mod rr_migrate;
pub mod saturation;
pub mod saturation_engine;
pub mod semantic_inventory;
pub mod spatial;
#[cfg(feature = "s2")]
pub mod spatial_s2;
pub mod standing;
pub mod state;
pub mod statement;
pub mod store;
pub mod stream;
pub mod symbolic;
pub mod sync_merge;
pub mod sync_scope;
pub mod tensor_derivation;
pub mod token_encoder;
pub mod training_export;
pub mod vector_batch;
pub mod vector_eval;
pub mod vector_payload;
pub mod versioned_graph;
pub mod working_log;
pub mod zerocopy;

pub use theorem_oplog::DeploymentId;

pub use access_method::{
    AccessMethod, AccessMethodCache, AccessMethodRegistry, AccessMethodRow, AccessMethodStats,
    AmResult, ColumnId, Cost, ModalityResolver, NoModalityResolver, OrderedAccessMethod, Predicate,
    PredicateMode, RankOutcome, RankedRow, RankingAccessMethod, RankingRegistry, RegionRef,
    RelationId, RowChange, RowChangeKind, RowId, RowIdStream, ScalarBound, ScalarValue,
    TimeSeriesAccessMethod,
};
pub use adaptive_index::{
    reconstruct_temporal_context, scalar_i64, ColdFragmentSkipMetadata, ColdSkipIndexDefinition,
    GraphStructuralIndexDefinition, GraphStructuralIndexKind, GraphStructuralIndexReceipt,
    SpatialIndexBackend, SpatialIndexDefinition, TemporalContextSlice, TemporalIndexDefinition,
};
pub use analogy::{
    accrete_schema, align_structure_mapping, downgrade_invalid_transfer, extract_schema,
    project_transfer_hypotheses, relational_signature, retrieve_relational_analogies,
    write_analogical_hypothesis_shadow, write_schema_record, AnalogicalTransferHypothesis,
    RelationalSignature, RelationalSignatureEdge, RelationalSignatureMatch,
    RelationalSignatureNode, RelationalSubgraph, SchemaRecord, SchemaRegistry, StructureMapping,
    ANALOGICAL_HYPOTHESIS_LABEL, ANALOGICAL_SCHEMA_INSTANCE_EDGE, ANALOGICAL_SCHEMA_INSTANCE_LABEL,
    ANALOGICAL_SCHEMA_LABEL, ANALOGICAL_USES_SCHEMA_EDGE,
};
pub use analogy_pass::{
    run_analogy_pass, AnalogyPassConfig, AnalogyPassReceipt, ANALOGICAL_MAPPING_LABEL,
    ANALOGY_PASS_BASE_CONFIDENCE, ANALOGY_PASS_MIN_STRUCTURAL_SCORE,
};
pub use assert_claim::{
    assert_claim, claim_id, is_asserted_claim, AssertClaimInput, AssertClaimReceipt,
    AssertClaimStore, ASSERTION_LABEL, ASSERTION_RECEIPT_LABEL, ASSERT_CLAIM_ENVIRONMENT,
    ASSERT_CLAIM_OPERATOR_ID, CLAIM_LABEL, RECEIPT_FOR_EDGE, SUPPORTED_BY_EDGE,
};
pub use chunk_store::{
    canonical_chunk_store_commit_hash, canonical_chunk_store_commit_preimage,
    validate_catalog_root, CatalogCommitResult, ChunkStore, ChunkStoreCatalog,
    ChunkStoreCompaction, ChunkWriteReport, CommitMeta, CommittedRoot, CompactionReport,
    MemChunkStore, RootCatalog, DOMAIN_ROOT_CATALOG, ROOT_NAME_DOCUMENTS, ROOT_NAME_GRAPH,
    ROOT_NAME_OBJECTS, ROOT_NAME_REDCORE,
};
pub use cold_fragments::{
    ColdFragment, ColdFragmentStore, CompressionFilter, FragmentColumn, FragmentRangeResult,
    FragmentRangeStats, HotColumnEncoding, HotColumnGroup, PromotionOutcome, PromotionPolicy,
    ZoneMap, EPISTEMIC_COLUMN_GROUP, EPISTEMIC_COLUMN_NAMES,
};
pub use cold_index::{
    ColdIndex, ColdIndexEntry, ColdScopeEntry, ColdTierKind, DiskColdIndex, InMemoryColdIndex,
    OrderedColdIndex,
};
pub use commands::{ThgCommand, ThgRequest, ThgResponse};
pub use context_view::{ContextView, ContextViewType, FreshnessStatus, HydrationHandle};
pub use crdt::{
    diff_since, join_delta, ActorId, Hlc, HlcClock, JoinReport, StampedBatch, StampedMutation,
    VersionVector,
};
pub use data_api::{
    DataCursor, DataDegradedState, DataFilter, DataFilterOperator, DataHydrate, DataLink,
    DataProvenance, DataQuery, DataRankSignals, DataRecord, DataResult, DataScope, DataSort,
    DataSortDirection, DataTrace, DEFAULT_DATA_QUERY_LIMIT, MAX_DATA_QUERY_LIMIT,
};
pub use data_registry::{
    conflict_edge as data_registry_conflict_edge, data_registry_entry_id,
    publication_edges as data_registry_publication_edges, DataRegistryConflict,
    DataRegistryContractError, DataRegistryEntry, DataRegistryIdentityBinding,
    DataRegistryObligation, DataRegistryObligationStatus, DataRegistryPublicationReceipt,
    DataSourceManifest, DataSourceManifestInput, DATA_REGISTRY_CONFLICT_LABEL,
    DATA_REGISTRY_ENTRY_LABEL, DATA_REGISTRY_RECEIPT_LABEL, DATA_SOURCE_MANIFEST_LABEL,
    MAX_DATA_REGISTRY_COLLECTIONS, MAX_DATA_REGISTRY_LOCATOR_BYTES,
    MAX_DATA_REGISTRY_METADATA_BYTES, MAX_DATA_REGISTRY_OBLIGATIONS,
    MAX_DATA_REGISTRY_SOURCE_ID_BYTES,
};
pub use derivation::{
    dependent_values, fold, fold_json, get_derivation, lift_rule_firing, lift_stream_event,
    put_assumption, put_derivation, record_nogood, rederive_if_missing, retract_assumption, why,
    AssumptionRecord, BooleanProvenance, BooleanSemiring, CountSemiring, DerivationAnnotation,
    DerivationGraphReader, DerivationGraphWriter, DerivationKind, DerivationRecord,
    DerivationSemiring, DerivationSemiringKind, DerivationSpec, FunctionalDerivationExecutor,
    NogoodRecord, ProbabilitySemiring, TropicalSemiring, WhyAssumption, WhyDerivation, WhyTree,
    ASSUMPTION_LABEL, CONFLICTS_WITH_EDGE, CONSUMES_EDGE, DERIVATION_LABEL, NOGOOD_LABEL,
    OPERATOR_LABEL, PRODUCES_EDGE, UNDER_EDGE, VALUE_LABEL, VIA_EDGE,
};
pub use doc_tree::{DocEntry, LegacyDocTree, LegacyDocTreeFixture, PathKey, PathPrefix};
pub use dynamic_ppr::{
    dynamic_ppr_full_reference, ppr_vectors_within_epsilon, DynamicPprEdgeUpdate,
    DynamicPprMaintainer, DynamicPprParams, DynamicPprStats, DynamicPprUpdateKind,
};
pub use epistemic::{
    compile_user_subgraph, epistemic_egraph_dedup, epistemic_shadow_edge_id,
    epistemic_shadow_node_id, epistemic_shadow_ppr, has_epistemic_shadow_edge_id,
    read_epistemic_shadow, read_same_eclass, run_epistemic_cron_pass, same_eclass_edge_id,
    select_nli_pairs, structural_epistemic_pass, try_read_epistemic_shadow, ConnectionFeatures,
    ConnectionScore, ConnectionScorer, EpistemicAnnotation, EpistemicAnnotations,
    EpistemicCandidatePair, EpistemicChokepoint, EpistemicCongruence, EpistemicCronInput,
    EpistemicCronReport, EpistemicDedupConfig, EpistemicDedupReport, EpistemicEnricher,
    EpistemicEnrichmentError, EpistemicEnrichmentMode, EpistemicEquivalenceClass,
    EpistemicFieldProvenance, EpistemicReadout, EpistemicRelationInput, EpistemicRelationKind,
    EpistemicRelationReadout, EpistemicShadowReadout, EpistemicSourceKind, GroundedExtensionStatus,
    LearnedConnectionScorer, LearnedConnectionScorerConfig, LearnedConnectionScorerPair,
    LearnedConnectionScorerRequest, LearnedConnectionScorerResponse, NliClassifier,
    NliEpistemicEnricher, NliPairInput, NliPairSelectionConfig, NliVerdict, PredictedEdgePointer,
    SameEClassRef, SourceReliability, StructuralEpistemicConfig, StructuralEpistemicInput,
    UserSubgraph, DEFAULT_CONNECTION_CALIBRATION_VERSION, DEFAULT_CONNECTION_FEATURE_VERSION,
    DEFAULT_CONNECTION_SCORER_MODEL_ID, DEFAULT_EPISTEMIC_ENGINE_VERSION, DEFAULT_NLI_MODEL_ID,
    EGRAPH_EPISTEMIC_ENGINE, EPISTEMIC_DETERMINISTIC_FALLBACK_ENV,
    EPISTEMIC_SCORER_CALIBRATION_ENV, EPISTEMIC_SCORER_ENDPOINT_ENV, EPISTEMIC_SCORER_MODEL_ENV,
    EPISTEMIC_SHADOW_LABEL, EPISTEMIC_SUPPORTS, HAS_EPISTEMIC_SHADOW, LEARNED_EPISTEMIC_ENGINE,
    NLI_EPISTEMIC_ENGINE, SAME_ECLASS, STRUCTURAL_EPISTEMIC_ENGINE, UNDERCUTS,
};
pub use epistemic::{
    is_structural_epistemic_edge_type, is_valued_epistemic_edge_type, valued_epistemic_edge_types,
};
pub use errors::{ThgError, ThgResult};
pub use events::{
    control_events_from_projections, project_node_mutation, ControlEventRecord, EventCompileError,
    EventOperation, MetadataOperation, MetadataSelector, NodeMutationProjection, ObjectSelector,
    RecordEvent, CONTROL_EVENT_LABEL, FIELD_SPEC_LABEL, METADATA_FIELD_SPEC_ENTITY,
    METADATA_OBJECT_TYPE_ENTITY, OBJECT_TYPE_LABEL as EVENT_OBJECT_TYPE_LABEL,
};
pub use executor::{execute_request_json, InMemoryThgExecutor, ThgExecutor};
pub use feature_dsl::{
    eval_feature, feature_score, malicious_probe_expressions, ArithOp, CompareOp, DynamicFeature,
    DynamicFeatureStatus, EvalResult, EvalSentinel, EvalValue, FeatureEvalBudget,
    FeatureEvalContext, FeatureExpr, TraversalTarget, TraverseKind, DEFAULT_MAX_AST_DEPTH,
    DEFAULT_MAX_STEPS, DEFAULT_MAX_TRAVERSAL_DEPTH, DEFAULT_MAX_TRAVERSAL_NODES,
};
pub use federated_blocks::{
    publish_visibility_invalidation, CustodyMergeReceipt, DiscoveryCandidate, FederatedGraphStore,
    FederatedPresence, FederatedVisibility, ForeignBlockRecord, ForeignBlockResolver,
    ForeignCacheEntry, ForeignOriginBinding, ForeignReadDepth, ForeignVerificationReceipt,
    IntroductionRecord, InvalidationReceipt, VerifiedForeignBlock, VisibilityInvalidation,
    BLOCK_PRESENCE_LABEL, CUSTODY_EVENT_LABEL, DISCOVERY_OPT_IN_LABEL, FOREIGN_CACHE_LABEL,
    INTRODUCTION_LABEL, VISIBILITY_INVALIDATION_KIND,
};
pub use file_payload::{
    decode_file_manifest, install_file_manifest_reference, read_file_payload, stage_file_payload,
    FileChunkRef, FileManifest, FileManifestRef, FilePayloadStagingGuard, FilePayloadWriteReport,
    FILE_CHUNK_MAX_BYTES, FILE_CHUNK_MIN_BYTES, FILE_CHUNK_TARGET_BYTES,
    FILE_CONTENT_MANIFEST_PROPERTY, FILE_HISTORY_ELIDED_COUNT_PROPERTY,
    FILE_PREVIOUS_CONTENT_MANIFESTS_PROPERTY, INLINE_FILE_BODY_PROPERTIES,
    MAX_FILE_MANIFEST_CHUNKS, MAX_PREVIOUS_FILE_MANIFESTS,
};
pub use fold::algebra::{
    AlgebraFold, AlgebraLawIdentity, AlgebraLeafCatalog, AlgebraOperationProfile,
    AlgebraProjection, AlgebraQuery, AlgebraReachability, AlgebraResidency, AlgebraStats,
    AlgebraValuationProfile, CalibrationEpoch, CsrView, EmbeddingExclusion,
    GraphBlasOperatorPairIdentity, LeafDtype, LeafKey, LeafRole, LeafSource, LeafSpec, LeafValues,
    LeafVector, MemTensorKind, NodeIndex, RowEmbedding, TokenCodes, TokenRegion, TokenResidency,
    TokenSlot, ALGEBRA_FOLD_VERSION, TOKEN_BIT_WIDTH, TURBOVEC_ENCODE_VERSION,
};
pub use fold::documents::{
    DocumentsFold, DocumentsFoldRecord, DocumentsQuery, DOCUMENTS_FOLD_VERSION,
    DOCUMENTS_INTENT_NAMESPACE, DOCUMENTS_INTENT_VERSION, FOLD_ID as DOCUMENTS_FOLD_ID,
    INTENT_PROVENANCE_SPAN_APPENDED, INTENT_RECORD_CREATED, INTENT_REVISION,
    INTENT_SKILL_EDITION_SELECTED, INTENT_SNAPSHOT_COMMITTED,
};
pub use fold::history::{
    HistoryCheckout, HistoryFold, HistoryInterval, HistoryQuery, HISTORY_FOLD_VERSION,
};
pub use fold::{
    ApplyReport, CheckpointRejectCode, CheckpointRejectionReceipt, ClientPin, Fold,
    FoldCapabilities, FoldCheckpoint, FoldDescriptor, FoldError, FoldId, FoldRuntime, FoldVersion,
    LogFold, RestoreDisposition, SegmentSummary, SegmentSummaryDescriptor, SegmentTransition,
    TransitionComposition, FOLD_CHECKPOINT_PROTOCOL_VERSION,
};
pub use fold_quarantine::{FoldConsumeError, FoldRefusal, FoldRefusalCode, QuarantineReceipt};
#[cfg(feature = "fjall-floor")]
pub use fold_storage::FjallFoldStorage;
#[cfg(feature = "redb-floor")]
pub use fold_storage::RedbFoldStorage;
pub use fold_storage::{
    FoldStorage, FoldStorageCommit, FoldStorageEntry, FoldStorageError, FoldStorageFloor,
    FoldStorageMutation, FOLD_STORAGE_FORMAT_VERSION,
};
pub use fulltext::{
    make_fulltext_backend, make_fulltext_backend_from_value, make_persistent_fulltext_backend,
    FieldedFullTextDefinition, FieldedFullTextDocument, FieldedFullTextHit, FieldedFullTextIndex,
    FullTextBackend, FullTextBackendDiagnostics, FullTextBackendError, FullTextDesignation,
    FullTextIndex, FullTextSearchBackend, FullTextSnippet, RUSTY_RED_FULLTEXT_BACKEND_ENV,
};
pub use graph::attribution::{GraphMemoryAttribution, GraphStats};
pub use graph::capability::{
    GraphAtomic, GraphRead, GraphResidency, GraphSnapshotSource, GraphStore, GraphTtl,
    GraphVectorPayloadAccess, GraphVerify, GraphWrite,
};
pub use graph::error::{GraphStoreError, GraphStoreResult};
pub use graph::fold_store::{
    FactTuple, FactsFold, FactsQuery, FoldGraphStore, FoldGraphTransaction, FoldMaintenanceFailure,
    GraphFold, GraphFoldQuery, GraphIntentBatch, GraphSemanticChange, PinnedGraphRead,
    PointReadExecution, PositionPinned, GRAPH_INTENT_NAME, GRAPH_INTENT_NAMESPACE,
    GRAPH_INTENT_VERSION,
};
pub use graph::footprint::{
    graph_memory_recompute_invocations, reset_graph_memory_recompute_invocations,
};
pub use graph::in_memory::InMemoryGraphStore;
#[allow(deprecated)]
pub use graph::louvain_communities;
pub use graph::mutation::{
    ConditionalNodePropertyUpdate, GraphMutation, GraphMutationBatch, GraphTransaction,
    GraphWriteResult, RecordedRedCoreMutation, RecordedRedCoreMutationBatch,
};
pub use graph::query::{NeighborHit, NeighborQuery, NodeQuery};
pub use graph::record::{
    Direction, EdgeRecord, EpistemicType, NodeRecord, Provenance, CALIBRATION_EPOCH_LABEL,
    EMBEDDING_EXCLUDED_PROPERTY, EMBEDDING_PROPERTY, VALUATION_PROPERTY,
};
pub use graph::snapshot::GraphSnapshot;
pub use graph::temporal::{edge_time_interval, TimeInterval};
pub use graph::ttl::{node_is_expired, node_ttl_expires_at_ms, now_ms, TTL_PROPERTY};
pub use graph::vector_index::{
    default_hybrid_edge_type_weights, HybridScoringConfig, VectorDesignation, VectorIndex,
    VectorPoint,
};
pub use graph::verify::{GraphRebuildReport, VerifyProblem, VerifyReport};
pub use graph::{
    connected_components, expand_bounded, expand_bounded_weighted, label_propagation_communities,
    pagerank, paths_shortest, paths_shortest_weighted, personalized_pagerank,
    personalized_pagerank_with_stats, EdgeTuple, PprPushStats,
};
pub use graph_csr::CsrGraph;
pub use graph_store::{
    graph_delta_commit_publications, graph_staged_commit_publications, manifest_version_compatible,
    read_manifest, reset_graph_commit_publication_counts, sanitize_tenant_segment, unix_ms,
    RedCoreDurability, RedCoreGraphStore, RedCoreManifest, RedCoreOptions, RedCoreStatus,
    CURRENT_FORMAT_VERSION, REDCORE_JOURNAL_FILE,
};
#[cfg(feature = "redis-store")]
pub use graph_store::{RedisGraphKeyspace, RedisGraphStore};
pub use hash::Hash;
pub use history_object_store::{
    publish_history_interval, ChunkPublishDisposition, ChunkPublishReceipt, PublishByChunkHash,
    S3CompatibleChunkPublisher,
};
pub use intern::{InternerStats, Symbol};
pub use journal::{JournalChunkStore, JournalCommitPolicy};
pub use layer::{
    LayerCardinality, LayerDescriptor, LayerDirection, LayerDurability, LayerId, LayerRegistry,
};
pub use motif::{
    extract_motifs, DomainSource, ExtractedMotif, MotifConfig, MOTIF_ADMITTED_RELATIONS,
    MOTIF_MAX_EDGES, MOTIF_MAX_NODES, MOTIF_RADIUS, MOTIF_SEED_LIMIT,
};
pub use partial_fold::{
    EvictionNotice, FoldSurface, MaterializationEligibility, MaterializedRead, MutationOrder,
    PartialFoldAccounting, PartialFoldError, PartialFoldRuntime, PartialKey, PartialOperator,
    PartialResidency, PositionedFoldIndex, PositionedSnapshot, ReconstructionStart,
    ReconstructionTicket, UpdateDisposition, PARTIAL_FOLD_INDEX_VERSION,
};
pub use recovery_profile::{RecoveryProfile, RecoveryStageSample};
pub use tensor_derivation::{
    ensure_assumption_for_epoch, ensure_assumption_for_pin, publish_tensor_block,
    tensor_block_value_id, InMemoryTensorBlockPayloadStore, TensorBlockAssumptionRefs,
    TensorBlockDerivationStore, TensorBlockLineageInput, TensorBlockManifestLineage,
    TensorBlockPayloadStore, TensorBlockPublicationInput, TensorBlockPublicationReceipt,
    TENSOR_BLOCK_POINTER_EDGE, TENSOR_BLOCK_POINTER_LABEL, TENSOR_BLOCK_VALUE_LABEL,
};
pub use token_encoder::{CalibratedEncoder, EncodedTokens, TokenEncodeError};

pub use colt::{
    asof_nearest_preceding, emptiness_gate, enumerate_join_candidates, execute_binary_join,
    execute_generic_join, execute_mask_join, execute_triangle_binary, execute_triangle_generic,
    join_shape_key, ColtLevel, Emission, EmptinessReceipt, JoinSteeringDecision, JoinSteeringState,
    JOIN_CANDIDATE_COLT_BINARY, JOIN_CANDIDATE_COLT_GENERIC, JOIN_CANDIDATE_MASK,
    NATIVE_JOIN_ALGORITHM,
};
pub use corroboration::{
    attack_severity, criticism_survival, solve_acceptability, spearman_rho, test_burden,
    AttackGraph, CorroborationParams, CorroborationStats, ATTACK_EDGE_TYPES,
};
pub use derivation_bridge::{fold_graph, why_graph, GraphDerivationStore};
pub use hooks::{
    coalesce_per_id, incremental_topology_post_commit_report, substrate_sync_hook, CoalesceKeyFn,
    HookContext, HookDispatcher, HookDispatcherConfig, HookDispatcherStats, HookEmitter, HookError,
    HookHandler, HookOutcome, HookRegistration, HookStoreAccess, IncrementalTopologyHookReport,
    MutationCommitContext, MutationEdgeContext, MutationEdgeRecordContext, MutationEdgeTransition,
    MutationEvent, MutationKind, MutationMatcher, MutationNodeContext, MutationNodeTransition,
    MutationScopeContext, SubstrateSyncEvent, SubstrateSyncOutbox,
};
pub use identity_index::{
    IdentityIndex, IdentityIndexDefinition, IdentityIndexKey, IdentityInsertOutcome,
    IdentityProblemRecord, IdentityTarget,
};
pub use incremental_topology::{
    plan_incremental_topology_update, topology_changed_nodes, topology_snapshot,
    IncrementalTopologyPlan, TopologyMark, TopologySnapshot,
};
pub use index_advisor::{
    IndexAdvisor, IndexAdvisorConfig, IndexPainKind, IndexPainSignal, ReceiptCluster,
    ShadowValidationReport,
};
pub use index_manifest::{
    IndexBackend, IndexBuildStatus, IndexCoverage, IndexCreatedBy, IndexEvalReceipt, IndexKind,
    IndexManifest, IndexScope, IndexSegmentRef, ServingIndexView,
};
pub use index_proposal::{IndexProposal, IndexProposalStatus, IndexRiskLevel, PromotionThreshold};
pub use index_registry::IndexRegistry;
pub use instant_kg::{
    instant_kg_payload_delta, instant_kg_payload_manifest, instant_kg_status_payload,
    CodeKgEncodedFile, CodeKgManifest, EdgeExplanation, HarnessInstantKg, ImpactResult,
    InstantKgStatus, PprResult, SearchResult, SessionDelta, INSTANT_KG_DEFAULT_ENCODER_VERSION,
    INSTANT_KG_DEFAULT_INGEST_VERSION, INSTANT_KG_PROTOCOL_VERSION,
};
pub use judgment::{
    propose_goal_for_surprise_gap, score_interestingness, InterestingnessConfig,
    InterestingnessScore, JudgmentItem, ProposedGoal,
};
pub use labeled_training_run::{
    LabeledTrainingRun, TrainingExportStatus, TrainingLabel, TrainingLabelFamily, TrainingOutcome,
    TrainingTaskType, ValidatorResult,
};
pub use map_artifact::{MapArtifact, MapArtifactDiff, MapArtifactType, MapSection};
pub use memory_document::{query_memory_documents, MemoryDocumentLens, MemoryDocumentQuery};
pub use merge_registry::{
    MergeRegistry, MergeRegistryEntry, MergeRegistryResolution, MergeRegistryStrategy,
};
pub use narrative::{
    build_narrative_view, CommonPlaceArcItem, NarrativeBuildError, NarrativeEvent, NarrativeLink,
    NarrativeView,
};
pub use node_state::{NodeStateBlock, NodeStateChannel};
pub use object_store::content_hash_bytes;
pub use object_store::{
    ColdObjectStore, DiskObjectStore, InMemoryObjectStore, LoadedColdObject, LoadedColdPack,
    PrivateManifestSlot, StagedRawPayload, UnifiedChunkStore,
};
pub use ordered::{
    EvictionFrontier, OrderedDesignation, OrderedIndex, OrderedIndexRegistry, OrderedMember,
    OrderedMode, OrderedScore, ScopedOrderedEntry, ScopedOrderedIndex, ScopedOrderedIndexManifest,
    ScopedOrderedIndexRegistry,
};
pub use planner::{
    compile_graphql_selection, execute_bounded_federation_join, execute_query,
    execute_query_with_resolver, validate_query_ir, AccessPathTrace, FederatedRowProvider,
    FederationJoinQuery, FederationRelation, FusionPolicy, GraphqlJoinSelection, GraphqlSelection,
    JoinBacking, JoinKind, JoinPlanPoint, JoinPredicate, PlanStageTrace, PlanTrace, Projection,
    QueryIr, QueryOutputRow, QueryRelation, QueryResult, RankerTrace,
};
pub use plugin::{
    normalize_plugin_command, PluginCapability, PluginCapabilityKind, PluginExecutionOutput,
    PluginOperationContext, PluginOperationHandler, PluginOperationRegistration, PluginRegistry,
    RustyRedPlugin,
};
pub use ppr_cache::{
    apply_maintained_ppr_update, cached_csr_personalized_pagerank, cached_personalized_pagerank,
    cached_single_seed_personalized_pagerank, clear_maintained_ppr_cache, clear_scoped_ppr_cache,
    maintained_ppr_cache_len, maintained_ppr_scores, maintained_ppr_stats, merge_ppr_scores,
    register_maintained_ppr_entry, reset_scoped_ppr_execution_counters, scoped_ppr_cache_len,
    scoped_ppr_execution_counters, ScopedPprExecutionCounters,
};
pub use provenance_semiring::{
    bounded_recursive_semiring_derivation, ProvenanceSemiring, ProvenanceSemiringKind,
    RecursiveSemiringEdge, RecursiveSemiringError, RecursiveSemiringResult, SemiringAnnotation,
};
pub use query_receipt::{
    AccessPathReceipt, QueryExplain, QueryKind, QueryOutcomeLabel, QueryReceipt, ReceiptScope,
};
pub use ranking::{
    apply_cascade, apply_feature_scores, compute_term_match, CascadeOutcome, EpistemicGate,
    ExpandRankingMethod, FeatureRankingConfig, QueryContext, RankCandidate, RankedCandidate,
    RankingRule, TermMatch, TextRankingMethod, TypoConfig, VectorRankingMethod,
};
pub use reachability::{
    compact_committed_reachable, compact_committed_reachable_if_due,
    compact_committed_reachable_with_limits, evaluate_compaction_policy, walk_committed_reachable,
    walk_committed_reachable_with_limits, CompactionPolicy, CompactionPolicyDecision,
    CompactionPolicyState, CoreReachabilityCodec, PolicyCompactionOutcome, ReachabilityChild,
    ReachabilityCodec, ReachabilityCompactionReport, ReachabilityKind, ReachabilityLimits,
    ReachabilityReport, ReachabilityStore, MAX_REACHABILITY_DEPTH, MAX_REACHABILITY_VISITS,
    PRODUCTION_REACHABILITY_LIMITS,
};
pub use read_model_index::{
    CompositeIndex, CompositeIndexDefinition, CompositeIndexEntry, CompositeIndexKey,
    CoveringIndex, CoveringIndexDefinition, CoveringRow, PartialIndex, PartialIndexDefinition,
    PartialIndexEntry, PartialPredicateClause, PartialPredicateOp,
};
pub use refinery::{
    build_refinery_manifest, derive_refinery_claims, run_refinery_standing, sign_refinery_manifest,
    source_class_reliability_prior, verify_refinery_manifest_signature,
    verify_refinery_manifest_spans, write_refinery_document, write_web_refinery_crawl_summary,
    write_web_refinery_document, CertifiedClaim, CorpusDeclaration, CorpusFileEntry, DpsExport,
    ProvActivity, ProvAgent, ProvEntity, ProvExport, RefineryCertification,
    RefineryClaimDerivationInput, RefineryClaimDerivationReport, RefineryDocumentInput,
    RefineryManifest, RefineryManifestInput, RefinerySignature, RefinerySpanVerificationReport,
    RefineryStandingInput, RefineryStandingReport, SourceReliabilityStanding,
    VerificationQueueItem, WebRefineryCrawlSummaryInput, WebRefineryDocumentInput,
    REFINERY_CLAIM_LABEL, REFINERY_DOCUMENT_LABEL,
};
pub use relational::{
    ColumnSchema, NativeAuthPrincipalRecord, NativeBillingAccountRecord, NativeCatalog,
    NativeProjectRecord, NativeTenantRecord, RelationDefinition, RelationSchema, RelationSource,
    RelationalGraphMut, RelationalLens, ResolvedRelation, ResolvedRow,
};
pub use revisable::{
    draft_from, publish, publish_if_head, supersede, PublishReceipt, Revisable, RevisionEnvelope,
    RevisionError, RevisionId, RevisionState, REVISION_LABEL, REVISION_OF_LINEAGE,
    SUPERSEDES_REVISION,
};
pub use saturation::{
    coalesce_by_subgraph, differential_check, facts_from_payload, facts_from_subgraph,
    materialize_closure, run_saturation, saturation_handler, saturation_hook_registration,
    validate_egglog_program, DifferentialReport, SaturationBackend, SaturationClosure,
    SaturationConfig, SaturationContributor, SaturationDerivedStatement,
    SaturationEquivalenceClass, SaturationFacts, SaturationPlugin, SaturationProgram,
    SaturationReport, SaturationRevisionReport, SaturationTemplateId,
    SATURATION_DERIVED_STATEMENT_LABEL, SATURATION_DERIVES_EDGE, SATURATION_ENGINE,
    SATURATION_ENGINE_VERSION, SATURATION_SHARED_RULE_IDS,
};
pub use spatial::{
    make_spatial_backend, make_spatial_backend_from_value, SpatialBackend, SpatialDesignation,
    SpatialError, SpatialIndex, RUSTY_RED_SPATIAL_BACKEND_ENV,
};
pub use standing::{
    StandingCommitError, StandingDefinition, StandingDelivery, StandingEvaluationError,
    StandingEvaluationMarker, StandingEvaluationReceipt, StandingEvaluator, StandingFiring,
    StandingFiringKind, StandingMatch, StandingQuery, StandingQueryError, StandingResult,
    StandingScope,
};
pub use state::{stable_hash, ThgEdge, ThgNode, ThgState};
pub use statement::{
    canonical_entity_id, collapse_if_corroborated, flatten_statements, literal_ref,
    migrate_epistemic_shadows_to_statements, predicate_id, predicate_incidence_edge,
    predicate_node, promote_statement_predicate, propose_same_as, statement_id,
    statement_incidence_edge_id, statement_incidence_edges, write_statement, Confidence,
    EpistemicShadowStatementMigrationReport, FlatObject, FlatTriple, StatementFieldProvenance,
    StatementProvenance, StatementQuery, StatementRecord, StatementSemiring, StatementWriteReceipt,
    CANONICAL_ENTITY_LABEL, HAS_OBJECT, HAS_PREDICATE, HAS_SUBJECT, PREDICATE_LABEL, SAME_AS,
    STATEMENT_LABEL,
};
pub use store::{InMemoryThgStore, ThgStore};
pub use stream::{
    AckPolicy, PendingDelivery, ReadDelivery, StreamDelta, StreamError, StreamEvent, StreamKey,
    StreamLog, StreamRegistry, StreamStore, StreamUrgency, DEFAULT_MAX_DELIVER,
    DEFAULT_READ_LIMIT as STREAM_READ_LIMIT, DEFAULT_REDELIVER_AFTER_MS,
};
pub use symbolic::{
    derive_datalog_receipt, derive_datalog_receipt_from_json, evolution_archive,
    evolution_archive_from_json, probabilistic_expected_value,
    probabilistic_expected_value_from_json, probabilistic_source_reliability,
    probabilistic_source_reliability_from_json, stable_hash_json, stable_hash_value,
    DATALOG_RULE_IDS,
};
pub use sync_merge::{
    list_conflict_shadows, merge_sync_snapshots, SyncConflictEntry, SyncMergeOutcome,
    SyncMergeSide, SYNC_CONFLICT_SHADOW_LABEL, SYNC_SHADOW_LOSER_SIDE_PROPERTY,
    SYNC_SHADOW_OF_PROPERTY, SYNC_SHADOW_REASON_PROPERTY,
};
pub use sync_scope::{
    exclude_local_only_from_snapshot, filter_snapshot_for_sync, is_connected_node,
    is_hosted_wins_node, is_local_only_node, is_synced_node, mark_local_only, node_transport,
    preserve_local_only_marker, tool_is_fresh_surface, transport_for_label, SyncDomainClass,
    SyncTransport, HOSTED_WINS_LABELS, LOCAL_ONLY_PROPERTY, OFFLINE_FRESH_SURFACE,
    SYNC_DOMAIN_CLASSES,
};
pub use training_export::{
    chat_samples_jsonl, export_records_jsonl, preference_samples_jsonl, sft_samples_from_records,
    ChatMessage, ChatSample, PreferenceSample, RedactionStatus, TrainingExportKind,
    TrainingExportRecord,
};
pub use vector_batch::{
    search_scoped_vector_batch, PersistentVectorBatchStore, ScopedVectorBatchReceipt,
    ScopedVectorPoint, ScopedVectorQuery,
};
pub use vector_eval::{
    filter_vector_candidates, vector_recall_against_exact, VectorFilterPolicy,
    VectorIndexDefinition, VectorRecallReport, VectorSearchBackend, VectorSearchCandidate,
    VectorSearchReceipt, VectorSearchWork,
};
pub use vector_payload::{
    decode_legacy_vector_array, read_vector_property, write_vector_property, VectorPropertyRef,
    VectorPropertyStagingGuard,
};
pub use versioned_graph::{
    apply_graph_mutation_batch, audit_published_prolly_parity, build_graph_payload_index,
    build_prolly_tree, build_prolly_tree_from_entries, build_prolly_tree_incremental,
    checkout_graph_version, compile_graph_pack, compile_graph_pack_incremental,
    compile_graph_pack_incremental_with_store, decode_canonical_graph_payload,
    delete_graph_ref_cas, diff_graph_snapshots, diff_graph_trees, edge_from_content_object,
    edge_to_content_object, graph_payload_address, graph_prolly_parity_validations,
    graph_version_log, hydrate_graph_snapshot_from_payload_index, legacy_graph_pack_v1_parents,
    merge_graph_snapshots, merge_graph_snapshots_with_registry,
    migrate_legacy_graph_content_object_v1, migrate_legacy_graph_pack_v1, node_from_content_object,
    node_to_content_object, object_bytes, prolly_validation_enabled,
    prolly_validation_sample_selected, prolly_validation_selected, recovered_graph_pack_descriptor,
    reset_graph_prolly_parity_validations, resolve_auto_confidence_edge,
    retarget_graph_pack_version, snapshot_content_objects, update_graph_payload_index,
    update_graph_ref, update_graph_ref_cas, validate_incremental_prolly_parity, CommitCost,
    CompiledGraphPack, GraphCheckoutResult, GraphCommit, GraphCompileOptions,
    GraphCompilerCapability, GraphContentObject, GraphDiffEntry, GraphMergeConflict,
    GraphMergeOptions, GraphMergeResolution, GraphMergeResult, GraphMergeSide, GraphMergeStrategy,
    GraphObjectKind, GraphPackManifest, GraphPayloadRecord, GraphProllyTree, GraphRefConflict,
    GraphRefDelete, GraphRefUpdate, GraphTreeChild, GraphTreeChunk, GraphTreeEntry, GraphTreeNode,
    GraphTreeParityAudit, GraphVersionDiff, GraphVersionLog, GraphVersionRef,
    GraphVersionRepository, IncrementalGraphPack, IncrementalTreeBuild, LegacyGraphObjectMigration,
    LegacyGraphPackMigration, ProllyReadMetrics, ProllyReader, StoredProllyNode,
    DEFAULT_GRAPH_BRANCH, DOMAIN_GRAPH_PAYLOAD, GRAPH_CHUNK_FORMAT_VERSION,
    GRAPH_PACK_COMPILER_VERSION, GRAPH_PAYLOAD_FORMAT_VERSION, PROLLY_VALIDATION_SAMPLE_SCALE_PPM,
    VERSIONED_GRAPH_PROTOCOL_VERSION,
};
pub use working_log::{
    RecencyCounter, RecencyState, TemporalFact, WorkingLog, WorkingLogEvent, WorkingLogEventKind,
};
pub use zerocopy::{
    access_archive, access_graph_archive, archive_content_object, archive_content_objects,
    archive_event_log, archive_hash, content_object_archive_bytes, edge_to_archive,
    mutation_to_archive, node_to_archive, replay_event_log, to_archive, tree_node_archive_bytes,
    ArchiveEdge, ArchiveGraphMutation, ArchiveGraphMutationLog, ArchiveNode, GraphArchiveBody,
    GraphArchiveContentObject, GraphArchiveEnvelope, GraphArchiveObjectBytes,
    GraphArchiveTreeChild, GraphArchiveTreeEntry, GraphArchiveTreeNode, MappedArchive,
    ZeroCopyArchiveError, GRAPH_ARCHIVE_FORMAT_VERSION, GRAPH_ARCHIVE_MIME_TYPE,
};

pub use turbovec::{IdMapIndex, TurboQuantIndex};

#[cfg(test)]
mod turbovec_pin_tests {
    #[test]
    fn turbovec_encode_version_matches_cargo_toml_pin() {
        let manifest = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"));
        let expected = format!(
            "turbovec = {{ version = \"={}\" }}",
            crate::TURBOVEC_ENCODE_VERSION
        );
        assert!(
            manifest.lines().any(|line| line.trim() == expected),
            "Cargo.toml must pin {expected}; TURBOVEC_ENCODE_VERSION={}",
            crate::TURBOVEC_ENCODE_VERSION
        );
    }
}
