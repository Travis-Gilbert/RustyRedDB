//! ArticleRead over retained RedCore roots. Reads are owned and fallible,
//! current scope admission is explicit, and historical visibility uses the
//! real compiler timestamp. No tenant snapshot or secondary graph is built.
use super::*;
use crate::article_fold::source::{
    derivation_evidence, exact_identity_matches, verified_source_admission,
};
use crate::article_fold::{
    ArticleFoldError, ArticleNode, ArticleRead, CitedEvidence, IdentityCandidates, PinnedSection,
    SourceAdmission,
};
use crate::graph::record::{Direction, EdgeRecord, NodeRecord};
use crate::graph::ttl::node_is_expired;
use std::collections::BTreeSet;
use theorem_surface_contracts::article::{ArticleCommit, HeadReference, HistoryCoverage, Revision};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommitMetadataVisibility {
    Visible,
    Redacted,
}

/// Presentation policy is mandatory at the native boundary. Redaction keeps
/// the genuine identity/root/transaction/time and never changes stored bytes.
pub fn project_article_commit(
    entry: &GraphHistoryEntry,
    visibility: CommitMetadataVisibility,
) -> Result<ArticleCommit, ArticleFoldError> {
    entry.validate().map_err(read_error)?;
    let redacted = visibility == CommitMetadataVisibility::Redacted;
    Ok(ArticleCommit {
        id: entry.commit.commit_hash.to_string(),
        tree: entry.commit.tree_hash.to_string(),
        transaction: Some(entry.roots.transaction),
        log_position: None,
        author: if redacted {
            String::new()
        } else {
            entry.commit.author.clone()
        },
        summary: if redacted {
            String::new()
        } else {
            entry.commit.message.clone()
        },
        metadata_redacted: redacted,
        timestamp_unix_ms: entry.commit.timestamp_unix_ms.to_string(),
    })
}

#[derive(Clone, Copy, Debug)]
pub enum ArticleRevisionTarget {
    Head,
    Commit(Hash),
    Before(Hash),
}

pub type PinnedSectionReader<'a> =
    dyn Fn(&PinnedGraphRevision, &str) -> Result<Vec<PinnedSection>, ArticleFoldError> + 'a;
pub type MetadataVisibilityPolicy<'a> = dyn Fn(
        &PinnedGraphHistory,
        &GraphHistoryEntry,
    ) -> Result<CommitMetadataVisibility, ArticleFoldError>
    + 'a;

pub struct RedCoreArticleReadContext<'a> {
    pub tenant: &'a str,
    /// Current authorization, including revocation, independent of old records.
    pub admission: &'a dyn Fn(&NodeRecord) -> bool,
    pub pins: &'a PinnedSectionReader<'a>,
    pub metadata_visibility: &'a MetadataVisibilityPolicy<'a>,
    pub seq: u64,
    pub stream_epoch: &'a str,
    /// A recorded server observation for live/legacy reads, never a revision id.
    pub observed_at_ms: i64,
}

pub struct RedCoreArticleRead<'a> {
    history: PinnedGraphHistory,
    revision: PinnedGraphRevision,
    context: RedCoreArticleReadContext<'a>,
    visible_at_ms: i64,
}

fn read_error(error: GraphStoreError) -> ArticleFoldError {
    ArticleFoldError::Read(error.to_string())
}

impl<'a> RedCoreArticleRead<'a> {
    pub fn new(
        history: &PinnedGraphHistory,
        target: ArticleRevisionTarget,
        context: RedCoreArticleReadContext<'a>,
    ) -> Result<Self, ArticleFoldError> {
        if context.tenant.trim().is_empty()
            || context.stream_epoch.trim().is_empty()
            || context.observed_at_ms < 0
        {
            return Err(ArticleFoldError::InvalidAuthority(
                "tenant, stream epoch and recorded visibility time are required".into(),
            ));
        }
        let history = match target {
            ArticleRevisionTarget::Head => history.clone(),
            ArticleRevisionTarget::Commit(commit) => history.view_at(commit).map_err(read_error)?,
            ArticleRevisionTarget::Before(commit) => history
                .before_commit(commit)
                .map_err(read_error)?
                .ok_or_else(|| ArticleFoldError::Missing("commit has no predecessor".into()))?,
        };
        let revision = history.head_revision().map_err(read_error)?;
        let visible_at_ms = if matches!(target, ArticleRevisionTarget::Head) {
            context.observed_at_ms
        } else {
            revision
                .timestamp_unix_ms()
                .map(|time| {
                    i64::try_from(time).map_err(|_| {
                        ArticleFoldError::InvalidAuthority(
                            "historical timestamp exceeds visibility clock range".into(),
                        )
                    })
                })
                .transpose()?
                .unwrap_or(context.observed_at_ms)
        };
        Ok(Self {
            history,
            revision,
            context,
            visible_at_ms,
        })
    }
    pub fn pinned_revision(&self) -> &PinnedGraphRevision {
        &self.revision
    }
    fn visible(&self, node: &NodeRecord) -> bool {
        !node.tombstone
            && !node_is_expired(node, self.visible_at_ms)
            && node
                .properties
                .get("status")
                .and_then(serde_json::Value::as_str)
                != Some("deleted")
    }
    fn scan_nodes(
        &self,
        mut visit: impl FnMut(NodeRecord) -> Result<bool, ArticleFoldError>,
    ) -> Result<(), ArticleFoldError> {
        let mut after = None;
        let mut count = 0;
        loop {
            let page = self
                .revision
                .records_page("node/", after.as_deref(), 256)
                .map_err(read_error)?;
            if page.is_empty() {
                return Ok(());
            }
            for record in page {
                after = Some(record.key());
                count += 1;
                if count > MAX_HISTORY_SCAN {
                    return Err(ArticleFoldError::IncompleteRead(
                        "pinned node scan exceeds work budget".into(),
                    ));
                }
                let GraphPayloadRecord::Node(node) = record else {
                    return Err(ArticleFoldError::Read("node scan returned an edge".into()));
                };
                if !visit(node)? {
                    return Ok(());
                }
            }
        }
    }
}

impl ArticleRead for RedCoreArticleRead<'_> {
    fn node(&self, id: &str) -> Result<ArticleNode, ArticleFoldError> {
        let Some(node) = self.revision.get_node(id).map_err(read_error)? else {
            return Ok(ArticleNode::Missing);
        };
        if !(self.context.admission)(&node) {
            return Ok(ArticleNode::Excluded);
        }
        if self.visible(&node) {
            Ok(ArticleNode::Present(node))
        } else {
            Ok(ArticleNode::Missing)
        }
    }
    fn edges(&self, id: &str, direction: Direction) -> Result<Vec<EdgeRecord>, ArticleFoldError> {
        let mut result = Vec::new();
        for edge in self
            .revision
            .incident_edges(id, direction)
            .map_err(read_error)?
        {
            if edge.tombstone {
                continue;
            }
            let mut excluded = false;
            for endpoint in [&edge.from_id, &edge.to_id] {
                if self
                    .revision
                    .get_node(endpoint)
                    .map_err(read_error)?
                    .is_some_and(|node| !(self.context.admission)(&node))
                {
                    excluded = true;
                    break;
                }
            }
            if excluded {
                if edge.from_id == id && crate::article_fold::is_persisted_collapse_edge(&edge) {
                    return Err(ArticleFoldError::Excluded);
                }
                continue;
            }
            result.push(edge);
            if result.len() > 10_000 {
                return Err(ArticleFoldError::IncompleteRead(
                    "adjacency exceeds article budget".into(),
                ));
            }
        }
        result.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(result)
    }
    fn identity_candidates(&self, exact: &str) -> Result<IdentityCandidates, ArticleFoldError> {
        let mut candidates = IdentityCandidates::default();
        self.scan_nodes(|node| {
            if !exact_identity_matches(&node, exact) {
                return Ok(true);
            }
            if !(self.context.admission)(&node) {
                candidates.excluded = true;
            } else if self.visible(&node) {
                candidates.nodes.push(node);
            }
            if candidates.nodes.len() > 10_000 {
                return Err(ArticleFoldError::IncompleteRead(
                    "identity candidate budget exceeded".into(),
                ));
            }
            Ok(true)
        })?;
        Ok(candidates)
    }
    fn source_admission(&self, source_ref: &str) -> Result<SourceAdmission, ArticleFoldError> {
        let mut result = SourceAdmission::Unresolved;
        self.scan_nodes(|node| {
            if !self.visible(&node) {
                return Ok(true);
            }
            // This exact signed receipt is authority about a source URL for
            // the receiving tenant, not article content. Requiring project
            // metadata here hides genuine foreign admissions as Unresolved.
            // The shared verifier checks receiving tenant, block URL, origin
            // binding, content hash and signature before classifying it.
            if let Some(admission) =
                verified_source_admission(&node, source_ref, self.context.tenant)?
            {
                let foreign = matches!(admission, SourceAdmission::Foreign { .. });
                result = admission;
                if foreign {
                    return Ok(false);
                }
            }
            Ok(true)
        })?;
        Ok(result)
    }
    fn evidence(&self, node: &NodeRecord) -> Result<Vec<CitedEvidence>, ArticleFoldError> {
        derivation_evidence(self, node)
    }
    fn pinned_sections(&self, canonical_id: &str) -> Result<Vec<PinnedSection>, ArticleFoldError> {
        (self.context.pins)(&self.revision, canonical_id)
    }
    fn revision(&self, dependency_keys: &BTreeSet<String>) -> Result<Revision, ArticleFoldError> {
        let commit = self
            .history
            .newest_touching(dependency_keys)
            .map_err(read_error)?
            .map(|entry| {
                let visibility = (self.context.metadata_visibility)(&self.history, &entry)?;
                project_article_commit(&entry, visibility)
            })
            .transpose()?;
        let roots = self.history.head_roots();
        let coverage = match self.history.coverage() {
            GraphHistoryCoverage::Complete => HistoryCoverage::Complete,
            GraphHistoryCoverage::Since {
                first_transaction,
                prior_history_unavailable,
            } => HistoryCoverage::Since {
                first_transaction,
                prior_history_unavailable,
            },
            GraphHistoryCoverage::Unavailable => HistoryCoverage::Unavailable,
        };
        Ok(Revision {
            commit,
            head: HeadReference {
                commit_id: Some(roots.logical_commit.to_string()),
                root: roots.logical_tree.root.to_string(),
                transaction: Some(roots.transaction),
                log_position: None,
            },
            coverage,
            seq: self.context.seq,
            stream_epoch: self.context.stream_epoch.into(),
        })
    }
}
