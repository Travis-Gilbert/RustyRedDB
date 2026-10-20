//! ArticleRead over an already-pinned HistoryFold projection. It borrows the
//! existing checkout once for a read session, never cloning it per article.
//! RedCore's production adapter implements ArticleRead directly with bounded,
//! fallible index reads; it does not construct this snapshot adapter.

use super::*;
use crate::fold::history::{HistoryCheckout, HistoryFold};
use crate::graph::snapshot::GraphSnapshot;
use crate::graph::ttl::node_is_expired;
use crate::hash::Hash;
use crate::versioned_graph::{
    canonical_commit_preimage, repository_object_at, GraphCommit, GraphContentObject,
};

/// A document adapter must resolve the document record, block and artifact at
/// the same graph revision. It may not use current DocStore hydration here.
pub type PinReader<'a> = dyn Fn(&str) -> Result<Vec<PinnedSection>, ArticleFoldError> + 'a;

pub struct HistoryArticleRead<'a> {
    history: &'a HistoryFold,
    checkout: HistoryCheckout,
    head: &'a GraphCommit,
    tenant: &'a str,
    /// Current admission is deliberately separate from historical membership.
    admission: &'a dyn Fn(&NodeRecord) -> bool,
    pins: &'a PinReader<'a>,
    seq: u64,
    stream_epoch: &'a str,
    visible_at_ms: i64,
}

impl<'a> HistoryArticleRead<'a> {
    pub fn new(
        history: &'a HistoryFold,
        position: theorem_oplog::LogPosition,
        tenant: &'a str,
        admission: &'a dyn Fn(&NodeRecord) -> bool,
        pins: &'a PinReader<'a>,
        seq: u64,
        stream_epoch: &'a str,
        visible_at_ms: i64,
    ) -> Result<Self, ArticleFoldError> {
        if tenant.trim().is_empty() || stream_epoch.trim().is_empty() {
            return Err(ArticleFoldError::InvalidAuthority(
                "tenant and stream epoch required".into(),
            ));
        }
        let checkout = history
            .checkout_at(position)
            .map_err(|e| ArticleFoldError::Read(e.to_string()))?
            .ok_or_else(|| {
                ArticleFoldError::IncompleteRead("history checkout unavailable".into())
            })?;
        let recorded = history
            .interval_at(checkout.interval.through)
            .ok_or_else(|| {
                ArticleFoldError::InvalidAuthority("history interval unavailable".into())
            })?;
        if recorded != &checkout.interval || checkout.snapshot.version != recorded.graph_version {
            return Err(ArticleFoldError::InvalidAuthority(
                "history checkout does not match pin".into(),
            ));
        }
        let head = history
            .repository()
            .commits
            .iter()
            .find(|commit| commit.commit_hash == recorded.commit_hash)
            .ok_or_else(|| {
                ArticleFoldError::IncompleteRead("actual history header is missing".into())
            })?;
        verify_commit(head)?;
        // HistoryFold's commit timestamp is a logical log-position value,
        // not wall time. Visibility is a separate explicit input captured by
        // the read authority; never reinterpret that position as Unix time.
        if visible_at_ms < 0 {
            return Err(ArticleFoldError::InvalidAuthority(
                "negative pinned visibility time".into(),
            ));
        }
        Ok(Self {
            history,
            checkout,
            head,
            tenant,
            admission,
            pins,
            seq,
            stream_epoch,
            visible_at_ms,
        })
    }

    fn snapshot(&self) -> &GraphSnapshot {
        &self.checkout.snapshot
    }

    fn visible(&self, node: &NodeRecord) -> bool {
        !node.tombstone
            && !node_is_expired(node, self.visible_at_ms)
            && string(node, "status") != Some("deleted")
    }

    /// Resolve one historical object through its addressed tree. No checkout
    /// or complete snapshot is cloned while examining a removed relationship.
    fn object_at(
        &self,
        position: theorem_oplog::LogPosition,
        key: &str,
    ) -> Result<Option<&GraphContentObject>, ArticleFoldError> {
        let interval = self.history.interval_at(position).ok_or_else(|| {
            ArticleFoldError::IncompleteRead("retained history interval missing".into())
        })?;
        let repository = self.history.repository();
        let pack = repository
            .packs
            .iter()
            .find(|pack| pack.commit.commit_hash == interval.commit_hash)
            .ok_or_else(|| {
                ArticleFoldError::IncompleteRead("retained history pack missing".into())
            })?;
        verify_commit(&pack.commit)?;
        if pack.tree.root != interval.tree_hash || pack.commit.tree_hash != interval.tree_hash {
            return Err(ArticleFoldError::InvalidAuthority(
                "retained tree pin mismatch".into(),
            ));
        }
        repository_object_at(repository, &pack.tree, key)
            .map_err(|e| ArticleFoldError::Read(e.to_string()))
    }

    fn changed_edge_touches(
        &self,
        index: usize,
        dependency_keys: &BTreeSet<String>,
    ) -> Result<bool, ArticleFoldError> {
        let interval = &self.history.intervals()[index];
        for key in interval
            .changed_object_keys
            .iter()
            .filter(|key| key.starts_with("edge/"))
        {
            let positions = std::iter::once(interval.through).chain(
                index
                    .checked_sub(1)
                    .map(|before| self.history.intervals()[before].through),
            );
            for position in positions {
                let Some(object) = self.object_at(position, key)? else {
                    continue;
                };
                let edge: EdgeRecord =
                    serde_json::from_value(object.payload.clone().ok_or_else(|| {
                        ArticleFoldError::IncompleteRead("retained edge payload missing".into())
                    })?)
                    .map_err(|e| ArticleFoldError::InvalidAuthority(e.to_string()))?;
                if ![&edge.from_id, &edge.to_id]
                    .into_iter()
                    .any(|id| dependency_keys.contains(&format!("node/{id}")))
                {
                    continue;
                }
                let mut allowed = true;
                for id in [&edge.from_id, &edge.to_id] {
                    if let Some(object) = self.object_at(position, &format!("node/{id}"))? {
                        let node: NodeRecord =
                            serde_json::from_value(object.payload.clone().ok_or_else(|| {
                                ArticleFoldError::IncompleteRead(
                                    "retained endpoint payload missing".into(),
                                )
                            })?)
                            .map_err(|e| ArticleFoldError::InvalidAuthority(e.to_string()))?;
                        allowed &= (self.admission)(&node);
                    }
                }
                if allowed {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
}

fn verify_commit(commit: &GraphCommit) -> Result<(), ArticleFoldError> {
    let bytes =
        canonical_commit_preimage(commit).map_err(|e| ArticleFoldError::Read(e.to_string()))?;
    if Hash::of(&bytes) != commit.commit_hash {
        return Err(ArticleFoldError::InvalidAuthority(
            "history commit hash mismatch".into(),
        ));
    }
    Ok(())
}

fn commit_projection(commit: &GraphCommit, log_position: u64) -> ArticleCommit {
    ArticleCommit {
        id: commit.commit_hash.to_prefixed(),
        tree: commit.tree_hash.to_prefixed(),
        transaction: None,
        log_position: Some(log_position),
        author: commit.author.clone(),
        summary: commit.message.clone(),
        metadata_redacted: false,
        timestamp_unix_ms: commit.timestamp_unix_ms.to_string(),
    }
}

impl ArticleRead for HistoryArticleRead<'_> {
    fn node(&self, id: &str) -> Result<ArticleNode, ArticleFoldError> {
        let Some(node) = self.snapshot().nodes.iter().find(|n| n.id == id) else {
            return Ok(ArticleNode::Missing);
        };
        if !(self.admission)(node) {
            return Ok(ArticleNode::Excluded);
        }
        if self.visible(node) {
            Ok(ArticleNode::Present(node.clone()))
        } else {
            Ok(ArticleNode::Missing)
        }
    }

    fn edges(&self, id: &str, direction: Direction) -> Result<Vec<EdgeRecord>, ArticleFoldError> {
        let mut result = Vec::new();
        for edge in &self.snapshot().edges {
            if edge.tombstone {
                continue;
            }
            let matches = match direction {
                Direction::Out => edge.from_id == id,
                Direction::In => edge.to_id == id,
            };
            if !matches {
                continue;
            }
            // Do not reveal an out-of-scope edge or its private endpoint. A
            // missing endpoint remains represented, for faithful red links.
            if [edge.from_id.as_str(), edge.to_id.as_str()]
                .into_iter()
                .any(|endpoint| {
                    self.snapshot()
                        .nodes
                        .iter()
                        .find(|n| n.id == endpoint)
                        .is_some_and(|n| !(self.admission)(n))
                })
            {
                if edge.from_id == id && super::is_persisted_collapse_edge(edge) {
                    return Err(ArticleFoldError::Excluded);
                }
                continue;
            }
            result.push(edge.clone());
            if result.len() > MAX_CLUSTER_READS {
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
        for node in &self.snapshot().nodes {
            let matched = source::exact_identity_matches(node, exact);
            if !matched {
                continue;
            }
            if !(self.admission)(node) {
                candidates.excluded = true;
                continue;
            }
            if self.visible(node) {
                candidates.nodes.push(node.clone());
            }
            if candidates.nodes.len() > MAX_CLUSTER_READS {
                return Err(ArticleFoldError::IncompleteRead(
                    "identity candidates exceed article budget".into(),
                ));
            }
        }
        Ok(candidates)
    }

    fn source_admission(&self, source_ref: &str) -> Result<SourceAdmission, ArticleFoldError> {
        let mut result = SourceAdmission::Unresolved;
        for node in &self.snapshot().nodes {
            if let Some(admission) =
                source::verified_source_admission(node, source_ref, self.tenant)?
            {
                if matches!(admission, SourceAdmission::Foreign { .. }) {
                    return Ok(admission);
                }
                result = admission;
            }
        }
        Ok(result)
    }

    fn evidence(&self, node: &NodeRecord) -> Result<Vec<CitedEvidence>, ArticleFoldError> {
        source::derivation_evidence(self, node)
    }

    fn pinned_sections(&self, canonical_id: &str) -> Result<Vec<PinnedSection>, ArticleFoldError> {
        (self.pins)(canonical_id)
    }

    fn revision(&self, dependency_keys: &BTreeSet<String>) -> Result<Revision, ArticleFoldError> {
        let mut interval = None;
        for (index, candidate) in self
            .history
            .intervals()
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, candidate)| candidate.through <= self.checkout.interval.through)
        {
            if candidate
                .changed_object_keys
                .iter()
                .any(|key| dependency_keys.contains(key))
                || self.changed_edge_touches(index, dependency_keys)?
            {
                interval = Some(candidate);
                break;
            }
        }
        let commit = interval
            .map(|interval| {
                let commit = self
                    .history
                    .repository()
                    .commits
                    .iter()
                    .find(|commit| commit.commit_hash == interval.commit_hash)
                    .ok_or_else(|| {
                        ArticleFoldError::IncompleteRead(
                            "retained cluster commit header missing".into(),
                        )
                    })?;
                verify_commit(commit)?;
                Ok::<_, ArticleFoldError>(commit_projection(commit, interval.through.0))
            })
            .transpose()?;
        Ok(Revision {
            commit,
            head: HeadReference {
                commit_id: Some(self.head.commit_hash.to_prefixed()),
                root: self.head.tree_hash.to_prefixed(),
                transaction: None,
                log_position: Some(self.checkout.interval.through.0),
            },
            coverage: HistoryCoverage::Complete,
            seq: self.seq,
            stream_epoch: self.stream_epoch.into(),
        })
    }
}
