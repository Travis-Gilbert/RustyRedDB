//! Append-only coordination streams for the Theorem HotGraph runtime.
//!
//! Streams are the live awareness channel for multi-head coordination. They
//! replace turn-start room polling with an append-only, cursor-read delta: an
//! actor pulls only the events on its subscribed streams that arrived after its
//! stored cursor. CRDT stays on the graph (shared *state*, the `crdt/` module);
//! streams carry *communication and awareness*.
//!
//! The primitive is the append-only special case of [`crate::ordered`]: where
//! `OrderedIndex` keys an `imbl::OrdMap` by score, a stream keys an
//! `imbl::OrdMap<u64, StreamEvent>` by a monotonic ordering token and reads the
//! tail after a cursor with the same range-after machinery:
//!
//! - `publish(stream_key, event) -> ordering_token` (append)
//! - `read_after(stream_key, cursor, limit) -> events` (OrdMap range `> cursor`)
//!
//! Cursors are per-`(actor, stream)` entries; subscriptions and pending pings
//! are per-`(tenant, actor)` sets so common actor names cannot leak events across
//! tenants. A ping (urgency `ask`|`block` with a `target_actor`) lands on the
//! stream like any event and additionally enqueues to the target's mention/wake
//! queue, bypassing the target's attention on purpose.
//!
//! Using `imbl` throughout mirrors [`crate::state::ThgState`]: embedding the
//! [`StreamStore`] there keeps the state clone copy-on-write (structural
//! sharing) and folds streams into the state hash like `runs`/`contexts`.
//!
//! Scope: every `stream_key` is `(tenant, topic)` resolved through the existing
//! [`sanitize_tenant_segment`] normalizer; an empty tenant is rejected rather
//! than routed to a default. No new scope resolver is introduced. The
//! coordination/MCP layer passes a tenant it already canonicalized through the
//! consolidation normalizer before calling in.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Bound;

use imbl::{OrdMap, OrdSet, Vector};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::graph_store::{sanitize_tenant_segment, unix_ms};

/// Default cap on the number of events returned by a single stream read.
pub const DEFAULT_READ_LIMIT: usize = 256;
/// Maximum retained events per stream. Older events are dropped on publish.
pub const MAX_STREAM_LOG_LEN: usize = 4096;
/// Maximum pending ping refs per `(tenant, actor)` wake queue.
pub const MAX_PENDING_PINGS_PER_ACTOR: usize = 256;

/// Separator between the tenant and topic segments inside a `stream_key`. `|`
/// cannot appear in a sanitized segment (`[A-Za-z0-9_-]`), so the join is
/// unambiguous.
const STREAM_KEY_SEPARATOR: char = '|';

/// A `(stream_key, ordering_token)` reference into the append-log. Pending pings
/// store these so the mention drain resolves each to its event in O(log n)
/// without scanning, while the log stays the single source of truth.
type EventRef = (String, u64);

/// Urgency dial: the passive-vs-active selector that already exists in the
/// coordination model. `Info` is passive (delta-read only). `Ask`/`Block` are
/// intentional pings that additionally reach the target's wake path.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamUrgency {
    #[default]
    Info,
    Ask,
    Block,
}

impl StreamUrgency {
    /// Parse an urgency token. An empty/absent value is the passive default
    /// (`Info`); anything unrecognized is rejected by the caller.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "info" => Some(Self::Info),
            "ask" => Some(Self::Ask),
            "block" => Some(Self::Block),
            _ => None,
        }
    }

    /// A ping is an intentional escalation that bridges to the wake/mention path.
    pub fn is_ping(self) -> bool {
        matches!(self, Self::Ask | Self::Block)
    }

    /// Alias used by the lower-level stream log API.
    pub fn pings(self) -> bool {
        self.is_ping()
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Ask => "ask",
            Self::Block => "block",
        }
    }
}

/// A single append-only coordination event.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StreamEvent {
    pub id: String,
    pub stream_key: String,
    pub ordering_token: u64,
    pub actor: String,
    pub kind: String,
    pub payload: Value,
    pub urgency: StreamUrgency,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_actor: Option<String>,
    pub created_at: String,
}

impl StreamEvent {
    /// A builder-light constructor. `ordering_token` is assigned by
    /// [`StreamLog::append`]; callers pass `0` and read the returned token.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: impl Into<String>,
        stream_key: impl Into<String>,
        actor: impl Into<String>,
        kind: impl Into<String>,
        payload: Value,
        urgency: StreamUrgency,
        target_actor: Option<String>,
        created_at: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            stream_key: stream_key.into(),
            ordering_token: 0,
            actor: actor.into(),
            kind: kind.into(),
            payload,
            urgency,
            target_actor: target_actor.filter(|value| !value.trim().is_empty()),
            created_at: created_at.into(),
        }
    }

    /// Whether this event is an intentional ping (`ask`/`block` with a target).
    pub fn is_ping(&self) -> bool {
        self.urgency.is_ping() && self.target_actor.is_some()
    }
}

/// Default backoff before an unacked event is redelivered (HANDOFF-HEAD-CALLS D2).
pub const DEFAULT_REDELIVER_AFTER_MS: u64 = 30_000;
/// Default delivery attempts before an event is dead-lettered.
pub const DEFAULT_MAX_DELIVER: u32 = 5;

/// How a read acknowledges the events it returns (JetStream-grade semantics,
/// HANDOFF-HEAD-CALLS D2).
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AckPolicy {
    /// Today's behavior: reading advances the cursor (ack-by-read). A consumer
    /// that crashes between read and action loses the event from its view.
    #[default]
    OnRead,
    /// The event stays pending until an explicit `ack`; an unacked event past
    /// `redeliver_after_ms` is redelivered on the next read with an incremented
    /// count, and after `max_deliver` attempts it is dead-lettered.
    Explicit,
}

/// One unacked in-flight delivery under [`AckPolicy::Explicit`].
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct PendingDelivery {
    pub ordering_token: u64,
    pub delivered_count: u32,
    pub delivered_at_ms: u64,
}

/// The outcome of a delivery-semantics read: the events handed to the consumer
/// this call (new plus redelivered), the resulting per-stream cursors, the
/// pending-delivery record for each delivered event, and any events that crossed
/// `max_deliver` and were dead-lettered this call (the caller writes the tension
/// record; the engine stays graph-write-free).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReadDelivery {
    pub events: Vec<StreamEvent>,
    pub cursors: OrdMap<String, u64>,
    pub pending: Vec<PendingDelivery>,
    pub dead_lettered: Vec<StreamEvent>,
}

/// A scope-resolution / validation failure. Carries a stable machine code that
/// the executor maps onto the [`crate::errors::ThgError`] envelope.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamError {
    pub code: &'static str,
    pub message: String,
}

impl StreamError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// The whole coordination substrate: all stream logs, per-`(actor, stream)`
/// cursors, per-`(tenant, actor)` subscription sets, and the per-target
/// pending-ping queue.
///
/// Every collection is an `imbl` persistent structure so embedding this in
/// [`crate::state::ThgState`] keeps state clones copy-on-write.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct StreamStore {
    /// Monotonic source of ordering tokens. Global across streams so every event
    /// id is unique and the per-stream subsequence stays strictly increasing --
    /// `read_after` ranges over tokens within a single stream.
    seq: u64,
    /// `stream_key` -> append-only log (`ordering_token` -> event).
    streams: OrdMap<String, OrdMap<u64, StreamEvent>>,
    /// `actor` -> (`stream_key` -> last consumed ordering token).
    cursors: OrdMap<String, OrdMap<String, u64>>,
    /// `(tenant, actor)` -> subscribed `stream_key`s (selective attention).
    subscriptions: OrdMap<String, OrdSet<String>>,
    /// `(tenant, target_actor)` -> queued ping references (the mention/wake bridge).
    pending_pings: OrdMap<String, Vector<EventRef>>,
    /// `actor` -> (`stream_key` -> unacked in-flight deliveries) under
    /// [`AckPolicy::Explicit`]. Empty under the default `OnRead` policy, and
    /// `serde(default)` so state persisted before D2 rehydrates cleanly.
    #[serde(default, skip_serializing_if = "OrdMap::is_empty")]
    pending: OrdMap<String, OrdMap<String, Vector<PendingDelivery>>>,
}

impl StreamStore {
    fn next_token(&mut self) -> u64 {
        self.seq = self.seq.saturating_add(1);
        self.seq
    }

    fn resolve_tenant_segment(tenant: &str) -> Result<String, StreamError> {
        let tenant_trimmed = tenant.trim();
        if tenant_trimmed.is_empty() {
            return Err(StreamError::new(
                "empty_tenant",
                "stream tenant must not be empty",
            ));
        }
        // Reuse the existing percent-encoding normalizer (`pct_` prefix). It is
        // reversible and never collapses a non-empty tenant to a default, so the
        // empty-tenant rejection above is the whole of the "no silent default"
        // contract.
        Ok(sanitize_tenant_segment(tenant_trimmed))
    }

    fn normalize_actor(actor: &str) -> Result<String, StreamError> {
        let trimmed = actor.trim();
        if trimmed.is_empty() {
            return Err(StreamError::new(
                "empty_actor",
                "stream actor must not be empty",
            ));
        }
        if trimmed.contains(STREAM_KEY_SEPARATOR) {
            return Err(StreamError::new(
                "invalid_actor",
                format!("stream actor must not contain '{STREAM_KEY_SEPARATOR}'"),
            ));
        }
        Ok(trimmed.to_string())
    }

    fn normalize_optional_actor(actor: Option<String>) -> Result<Option<String>, StreamError> {
        match actor {
            Some(raw) => {
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    Ok(None)
                } else {
                    Self::normalize_actor(trimmed).map(Some)
                }
            }
            None => Ok(None),
        }
    }

    fn actor_scope_key(tenant: &str, actor: &str) -> Result<String, StreamError> {
        let tenant_segment = Self::resolve_tenant_segment(tenant)?;
        let actor = Self::normalize_actor(actor)?;
        let actor_segment = sanitize_tenant_segment(&actor);
        Ok(format!(
            "{tenant_segment}{STREAM_KEY_SEPARATOR}{actor_segment}"
        ))
    }

    fn ensure_stream_key_belongs_to_tenant(
        tenant: &str,
        stream_key: &str,
    ) -> Result<(), StreamError> {
        let tenant_segment = Self::resolve_tenant_segment(tenant)?;
        let expected_prefix = format!("{tenant_segment}{STREAM_KEY_SEPARATOR}");
        if stream_key.starts_with(&expected_prefix) {
            Ok(())
        } else {
            Err(StreamError::new(
                "tenant_mismatch",
                "stream key does not belong to the requested tenant",
            ))
        }
    }

    /// Resolve `(tenant, topic)` into a `stream_key` via the existing tenant
    /// normalizer. An empty tenant is rejected, never defaulted; an empty topic
    /// is rejected. No new scope resolver is introduced.
    pub fn resolve_stream_key(tenant: &str, topic: &str) -> Result<String, StreamError> {
        let tenant_segment = Self::resolve_tenant_segment(tenant)?;
        let topic_trimmed = topic.trim();
        if topic_trimmed.is_empty() {
            return Err(StreamError::new(
                "empty_stream_topic",
                "stream topic must not be empty",
            ));
        }
        let topic_segment = sanitize_tenant_segment(topic_trimmed);
        Ok(format!(
            "{tenant_segment}{STREAM_KEY_SEPARATOR}{topic_segment}"
        ))
    }

    /// Append an event to a `(tenant, topic)` stream and return it (carrying its
    /// new ordering token). A ping (urgency `ask`|`block` with a `target_actor`)
    /// also enqueues to the target's pending-ping queue, regardless of whether
    /// the target is subscribed -- a ping bypasses attention by design.
    #[allow(clippy::too_many_arguments)]
    pub fn publish(
        &mut self,
        tenant: &str,
        topic: &str,
        actor: &str,
        kind: &str,
        payload: Value,
        urgency: StreamUrgency,
        target_actor: Option<String>,
    ) -> Result<StreamEvent, StreamError> {
        let stream_key = Self::resolve_stream_key(tenant, topic)?;
        let actor = Self::normalize_actor(actor)?;
        let target_actor = Self::normalize_optional_actor(target_actor)?;
        if urgency.is_ping() && target_actor.is_none() {
            return Err(StreamError::new(
                "missing_target_actor",
                "urgency ask|block requires a target_actor",
            ));
        }
        let token = self.next_token();
        let event = StreamEvent {
            id: format!("evt:{token:016x}"),
            stream_key: stream_key.clone(),
            ordering_token: token,
            actor,
            kind: kind.to_string(),
            payload,
            urgency,
            target_actor: target_actor.clone(),
            created_at: format!("unix_ms:{}", unix_ms()),
        };

        // Append to the per-stream log (copy-on-write clone of the inner OrdMap).
        let mut log = self.streams.get(&stream_key).cloned().unwrap_or_default();
        log.insert(token, event.clone());
        let mut evicted_tokens = Vec::new();
        while log.len() > MAX_STREAM_LOG_LEN {
            if let Some(oldest_token) = log.iter().next().map(|(token, _)| *token) {
                log.remove(&oldest_token);
                evicted_tokens.push(oldest_token);
            } else {
                break;
            }
        }
        self.streams.insert(stream_key.clone(), log);
        for evicted_token in evicted_tokens {
            self.prune_pending_ref(&stream_key, evicted_token);
        }

        if urgency.is_ping() {
            if let Some(target) = target_actor {
                let target_key = Self::actor_scope_key(tenant, &target)?;
                let mut queue = self
                    .pending_pings
                    .get(&target_key)
                    .cloned()
                    .unwrap_or_default();
                queue.push_back((stream_key, token));
                while queue.len() > MAX_PENDING_PINGS_PER_ACTOR {
                    queue.pop_front();
                }
                self.pending_pings.insert(target_key, queue);
            }
        }
        Ok(event)
    }

    fn prune_pending_ref(&mut self, stream_key: &str, token: u64) {
        let keys: Vec<String> = self.pending_pings.keys().cloned().collect();
        for actor_key in keys {
            let Some(queue) = self.pending_pings.get(&actor_key).cloned() else {
                continue;
            };
            let mut retained = Vector::new();
            for event_ref in queue {
                if !(event_ref.0 == stream_key && event_ref.1 == token) {
                    retained.push_back(event_ref);
                }
            }
            if retained.is_empty() {
                self.pending_pings.remove(&actor_key);
            } else {
                self.pending_pings.insert(actor_key, retained);
            }
        }
    }

    fn cursor_for(&self, actor: &str, stream_key: &str) -> u64 {
        self.cursors
            .get(actor)
            .and_then(|streams| streams.get(stream_key))
            .copied()
            .unwrap_or(0)
    }

    fn set_cursor(&mut self, actor: &str, stream_key: &str, token: u64) {
        let mut actor_cursors = self.cursors.get(actor).cloned().unwrap_or_default();
        actor_cursors.insert(stream_key.to_string(), token);
        self.cursors.insert(actor.to_string(), actor_cursors);
    }

    /// Events strictly after `cursor` in one stream, ascending by token, capped
    /// at `limit`. Reuses the OrdMap range-after machinery (`crate::ordered`):
    /// the log is token-ordered, so a half-open range from `cursor` is exact and
    /// O(result + log n) -- nothing in the window is missed or duplicated.
    fn read_after(log: &OrdMap<u64, StreamEvent>, cursor: u64, limit: usize) -> Vec<StreamEvent> {
        log.range((Bound::Excluded(cursor), Bound::Unbounded))
            .take(limit)
            .map(|(_, event)| event.clone())
            .collect()
    }

    /// The actor's current tenant-scoped subscription set, as resolved `stream_key`s.
    pub fn subscriptions_for(&self, tenant: &str, actor: &str) -> Result<Vec<String>, StreamError> {
        let actor_key = Self::actor_scope_key(tenant, actor)?;
        Ok(self
            .subscriptions
            .get(&actor_key)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default())
    }

    /// Pull events after the actor's cursor across `stream_keys` (or the actor's
    /// tenant-scoped subscription set when `stream_keys` is empty -- selective
    /// attention).
    /// Advances the per-`(actor, stream)` cursor when `advance` is set. Returns
    /// the events in a single total order plus the resulting cursor per stream.
    pub fn read(
        &mut self,
        tenant: &str,
        actor: &str,
        stream_keys: &[String],
        advance: bool,
        limit: usize,
    ) -> Result<(Vec<StreamEvent>, OrdMap<String, u64>), StreamError> {
        let limit = if limit == 0 {
            DEFAULT_READ_LIMIT
        } else {
            limit
        };
        let _ = Self::resolve_tenant_segment(tenant)?;
        let actor = Self::normalize_actor(actor)?;
        let keys: Vec<String> = if stream_keys.is_empty() {
            self.subscriptions_for(tenant, &actor)?
        } else {
            for stream_key in stream_keys {
                Self::ensure_stream_key_belongs_to_tenant(tenant, stream_key)?;
            }
            stream_keys.to_vec()
        };

        let mut events = Vec::new();
        for key in &keys {
            let cursor = self.cursor_for(&actor, key);
            events.extend(
                self.streams
                    .get(key)
                    .map(|log| Self::read_after(log, cursor, limit))
                    .unwrap_or_default(),
            );
        }
        // Global ordering tokens give a single total order across streams with no
        // merge step.
        events.sort_by_key(|event| event.ordering_token);
        events.truncate(limit);

        let mut new_cursors = OrdMap::new();
        for key in keys {
            let standing = self.cursor_for(&actor, &key);
            let token = events
                .iter()
                .filter(|event| event.stream_key == key)
                .map(|event| event.ordering_token)
                .max()
                .unwrap_or(standing);
            if advance && token != standing {
                self.set_cursor(&actor, &key, token);
            }
            new_cursors.insert(key, token);
        }
        Ok((events, new_cursors))
    }

    fn pending_for(&self, actor: &str, stream_key: &str) -> Vector<PendingDelivery> {
        self.pending
            .get(actor)
            .and_then(|streams| streams.get(stream_key))
            .cloned()
            .unwrap_or_default()
    }

    fn set_pending(&mut self, actor: &str, stream_key: &str, list: Vec<PendingDelivery>) {
        let mut actor_map = self.pending.get(actor).cloned().unwrap_or_default();
        if list.is_empty() {
            actor_map.remove(stream_key);
        } else {
            actor_map.insert(stream_key.to_string(), list.into_iter().collect());
        }
        if actor_map.is_empty() {
            self.pending.remove(actor);
        } else {
            self.pending.insert(actor.to_string(), actor_map);
        }
    }

    /// Delivery-semantics read (HANDOFF-HEAD-CALLS D2). `OnRead` reproduces
    /// [`Self::read`] with `advance=true` exactly. `Explicit` delivers new events
    /// after the cursor and records them pending; an unacked event past
    /// `redeliver_after_ms` is redelivered on a later read with an incremented
    /// count; an event that reaches `max_deliver` is dead-lettered (returned so
    /// the caller can write the tension record) and dropped from pending. The
    /// cursor still advances past new events -- they live in `pending` until
    /// [`Self::ack`], not on the cursor. `now_ms` is injected so redelivery
    /// timing is deterministic and testable.
    #[allow(clippy::too_many_arguments)]
    pub fn read_delivery(
        &mut self,
        tenant: &str,
        actor: &str,
        stream_keys: &[String],
        ack_policy: AckPolicy,
        now_ms: u64,
        redeliver_after_ms: u64,
        max_deliver: u32,
        limit: usize,
    ) -> Result<ReadDelivery, StreamError> {
        let limit = if limit == 0 {
            DEFAULT_READ_LIMIT
        } else {
            limit
        };
        let _ = Self::resolve_tenant_segment(tenant)?;
        let actor = Self::normalize_actor(actor)?;
        let keys: Vec<String> = if stream_keys.is_empty() {
            self.subscriptions_for(tenant, &actor)?
        } else {
            for stream_key in stream_keys {
                Self::ensure_stream_key_belongs_to_tenant(tenant, stream_key)?;
            }
            stream_keys.to_vec()
        };

        // OnRead: identical to the classic advancing read, no pending tracking.
        if ack_policy == AckPolicy::OnRead {
            let (events, cursors) = self.read(tenant, &actor, &keys, true, limit)?;
            return Ok(ReadDelivery {
                events,
                cursors,
                pending: Vec::new(),
                dead_lettered: Vec::new(),
            });
        }

        #[derive(Clone)]
        struct DeliveryCandidate {
            stream_key: String,
            event: StreamEvent,
            pending: PendingDelivery,
            is_new: bool,
        }

        let mut candidates: Vec<DeliveryCandidate> = Vec::new();
        let mut pending_by_stream: BTreeMap<String, Vec<PendingDelivery>> = BTreeMap::new();
        let mut pending_out: Vec<PendingDelivery> = Vec::new();
        let mut dead_lettered: Vec<StreamEvent> = Vec::new();
        let mut cursors_out: OrdMap<String, u64> = OrdMap::new();

        for key in &keys {
            let log = self.streams.get(key).cloned().unwrap_or_default();
            let mut next_pending: Vec<PendingDelivery> = Vec::new();
            let cursor = self.cursor_for(&actor, key);
            cursors_out.insert(key.clone(), cursor);

            // Sweep existing pending: dead-letter the exhausted, redeliver the due.
            for pd in self.pending_for(&actor, key) {
                let Some(event) = log.get(&pd.ordering_token).cloned() else {
                    dead_lettered.push(StreamEvent {
                        id: format!("evt:evicted:{:016x}", pd.ordering_token),
                        stream_key: key.clone(),
                        ordering_token: pd.ordering_token,
                        actor: "stream_store".to_string(),
                        kind: "evicted_before_ack".to_string(),
                        payload: json!({
                            "delivered_count": pd.delivered_count,
                            "delivered_at_ms": pd.delivered_at_ms,
                        }),
                        urgency: StreamUrgency::Info,
                        target_actor: Some(actor.clone()),
                        created_at: format!("unix_ms:{now_ms}"),
                    });
                    continue;
                };
                if pd.delivered_count >= max_deliver {
                    dead_lettered.push(event);
                    continue;
                }
                if now_ms.saturating_sub(pd.delivered_at_ms) >= redeliver_after_ms {
                    let bumped = PendingDelivery {
                        ordering_token: pd.ordering_token,
                        delivered_count: pd.delivered_count + 1,
                        delivered_at_ms: now_ms,
                    };
                    candidates.push(DeliveryCandidate {
                        stream_key: key.clone(),
                        event,
                        pending: bumped,
                        is_new: false,
                    });
                }
                next_pending.push(pd);
            }

            // New events after the cursor: deliver and record pending (count 1).
            for event in Self::read_after(&log, cursor, limit) {
                let pd = PendingDelivery {
                    ordering_token: event.ordering_token,
                    delivered_count: 1,
                    delivered_at_ms: now_ms,
                };
                candidates.push(DeliveryCandidate {
                    stream_key: key.clone(),
                    event,
                    pending: pd,
                    is_new: true,
                });
            }
            pending_by_stream.insert(key.clone(), next_pending);
        }

        candidates.sort_by_key(|candidate| candidate.event.ordering_token);
        let mut delivered: Vec<StreamEvent> = Vec::new();
        for candidate in candidates.into_iter().take(limit) {
            if candidate.is_new {
                pending_by_stream
                    .entry(candidate.stream_key.clone())
                    .or_default()
                    .push(candidate.pending.clone());
                let cursor = cursors_out
                    .get(&candidate.stream_key)
                    .copied()
                    .unwrap_or_default()
                    .max(candidate.event.ordering_token);
                cursors_out.insert(candidate.stream_key.clone(), cursor);
            } else if let Some(list) = pending_by_stream.get_mut(&candidate.stream_key) {
                if let Some(existing) = list
                    .iter_mut()
                    .find(|pd| pd.ordering_token == candidate.pending.ordering_token)
                {
                    *existing = candidate.pending.clone();
                }
            }
            pending_out.push(candidate.pending);
            delivered.push(candidate.event);
        }

        for key in &keys {
            let cursor = cursors_out.get(key).copied().unwrap_or_default();
            if cursor != self.cursor_for(&actor, key) {
                self.set_cursor(&actor, key, cursor);
            }
            self.set_pending(
                &actor,
                key,
                pending_by_stream.remove(key).unwrap_or_default(),
            );
        }
        dead_lettered.sort_by_key(|event| event.ordering_token);
        Ok(ReadDelivery {
            events: delivered,
            cursors: cursors_out,
            pending: pending_out,
            dead_lettered,
        })
    }

    /// Acknowledge delivered events by ordering token, removing them from the
    /// actor's pending set so they are not redelivered. Returns the number of
    /// pending entries cleared.
    pub fn ack(
        &mut self,
        tenant: &str,
        actor: &str,
        stream_key: &str,
        tokens: &[u64],
    ) -> Result<usize, StreamError> {
        Self::ensure_stream_key_belongs_to_tenant(tenant, stream_key)?;
        let actor = Self::normalize_actor(actor)?;
        let acked: BTreeSet<u64> = tokens.iter().copied().collect();
        let existing = self.pending_for(&actor, stream_key);
        let before = existing.len();
        let kept: Vec<PendingDelivery> = existing
            .into_iter()
            .filter(|pd| !acked.contains(&pd.ordering_token))
            .collect();
        let cleared = before - kept.len();
        self.set_pending(&actor, stream_key, kept);
        Ok(cleared)
    }

    /// Add `stream_key` to the actor's subscription set; returns the new set.
    ///
    /// Opting in starts attention *now*: a first-time subscriber's cursor is
    /// initialized to the stream's current head, so the next read returns events
    /// published after the subscription rather than the full backlog. A
    /// re-subscribe keeps any existing cursor, resuming where the actor left off.
    /// A future cursor-override API can expose historical replay explicitly.
    pub fn subscribe(
        &mut self,
        tenant: &str,
        actor: &str,
        stream_key: &str,
    ) -> Result<Vec<String>, StreamError> {
        Self::ensure_stream_key_belongs_to_tenant(tenant, stream_key)?;
        let actor_key = Self::actor_scope_key(tenant, actor)?;
        let actor = Self::normalize_actor(actor)?;
        let mut set = self
            .subscriptions
            .get(&actor_key)
            .cloned()
            .unwrap_or_default();
        set.insert(stream_key.to_string());
        self.subscriptions.insert(actor_key, set);

        let has_cursor = self
            .cursors
            .get(&actor)
            .map(|cursors| cursors.contains_key(stream_key))
            .unwrap_or(false);
        if !has_cursor {
            let head = self
                .streams
                .get(stream_key)
                .and_then(|log| log.iter().next_back().map(|(token, _)| *token))
                .unwrap_or(0);
            self.set_cursor(&actor, stream_key, head);
        }
        self.subscriptions_for(tenant, &actor)
    }

    /// Remove `stream_key` from the actor's subscription set; returns the new set.
    pub fn unsubscribe(
        &mut self,
        tenant: &str,
        actor: &str,
        stream_key: &str,
    ) -> Result<Vec<String>, StreamError> {
        Self::ensure_stream_key_belongs_to_tenant(tenant, stream_key)?;
        let actor_key = Self::actor_scope_key(tenant, actor)?;
        if let Some(mut set) = self.subscriptions.get(&actor_key).cloned() {
            set.remove(stream_key);
            if set.is_empty() {
                self.subscriptions.remove(&actor_key);
            } else {
                self.subscriptions.insert(actor_key, set);
            }
        }
        self.subscriptions_for(tenant, actor)
    }

    /// Drain (or peek) the target actor's pending pings, resolving each reference
    /// back to its event in arrival order. This is the seam the Stop-hook mention
    /// drain (warm head) and the spawn/courier wake (cold head) bridge onto.
    /// Delivery never clears the queue; callers acknowledge handled ordering
    /// tokens with [`Self::ack_mentions`] so a failed wake remains retryable.
    pub fn drain_mentions(
        &mut self,
        tenant: &str,
        actor: &str,
        _advance: bool,
    ) -> Result<Vec<StreamEvent>, StreamError> {
        let actor_key = Self::actor_scope_key(tenant, actor)?;
        let refs = match self.pending_pings.get(&actor_key) {
            Some(refs) if !refs.is_empty() => refs.clone(),
            _ => return Ok(Vec::new()),
        };
        let mut events: Vec<StreamEvent> = refs
            .iter()
            .filter_map(|(stream_key, token)| self.event_at(stream_key, *token))
            .collect();
        events.sort_by_key(|event| event.ordering_token);
        Ok(events)
    }

    pub fn ack_mentions(
        &mut self,
        tenant: &str,
        actor: &str,
        ordering_tokens: &[u64],
    ) -> Result<usize, StreamError> {
        let actor_key = Self::actor_scope_key(tenant, actor)?;
        if ordering_tokens.is_empty() {
            return Ok(0);
        }
        let acknowledged = ordering_tokens.iter().copied().collect::<BTreeSet<_>>();
        let Some(refs) = self.pending_pings.get(&actor_key).cloned() else {
            return Ok(0);
        };
        let retained = refs
            .iter()
            .filter(|(_, token)| !acknowledged.contains(token))
            .cloned()
            .collect::<Vector<_>>();
        let removed = refs.len().saturating_sub(retained.len());
        if retained.is_empty() {
            self.pending_pings.remove(&actor_key);
        } else {
            self.pending_pings.insert(actor_key, retained);
        }
        Ok(removed)
    }

    fn event_at(&self, stream_key: &str, token: u64) -> Option<StreamEvent> {
        self.streams
            .get(stream_key)
            .and_then(|log| log.get(&token))
            .cloned()
    }
}

/// The `(tenant, topic)` scope of a stream. `topic` is the room, optionally finer
/// grained (per-task, per-actor). The [`canonical`](StreamKey::canonical) string
/// is the opaque key used to address a [`StreamLog`] in a [`StreamRegistry`] and
/// to key durable storage.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct StreamKey {
    pub tenant: String,
    pub topic: String,
}

impl StreamKey {
    pub fn new(tenant: impl Into<String>, topic: impl Into<String>) -> Self {
        Self {
            tenant: tenant.into(),
            topic: topic.into(),
        }
    }

    /// The opaque canonical key. Uses a unit-separator join so a tenant or topic
    /// containing a literal separator can never forge a different scope.
    pub fn canonical(&self) -> String {
        format!("{}\u{1}{}", self.tenant, self.topic)
    }
}

/// A single append-only stream: events keyed by a strictly-increasing ordering
/// token. The append-only specialization of [`OrderedIndex`](crate::ordered::OrderedIndex).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StreamLog {
    by_token: OrdMap<u64, StreamEvent>,
    next_token: u64,
}

impl StreamLog {
    pub fn new() -> Self {
        Self {
            by_token: OrdMap::new(),
            next_token: 1,
        }
    }

    /// Append an event, assigning it the next ordering token (always `>` every
    /// token already in the log). Returns the assigned token. The event's
    /// `ordering_token` field is overwritten with the assignment.
    pub fn append(&mut self, mut event: StreamEvent) -> u64 {
        let token = self.next_token.max(1);
        event.ordering_token = token;
        self.by_token.insert(token, event);
        while self.by_token.len() > MAX_STREAM_LOG_LEN {
            let Some((oldest, _)) = self.by_token.iter().next() else {
                break;
            };
            let oldest = *oldest;
            self.by_token.remove(&oldest);
        }
        self.next_token = token.saturating_add(1);
        token
    }

    /// Re-insert an event at its own recorded token (durable rehydration). Does
    /// not assign a new token; keeps `next_token` ahead of every ingested token
    /// so a later [`append`](Self::append) stays monotonic.
    pub fn ingest(&mut self, event: StreamEvent) {
        let token = event.ordering_token;
        self.next_token = self.next_token.max(token.saturating_add(1));
        self.by_token.insert(token, event);
    }

    /// Events strictly after `cursor`, in ascending token order, up to `limit`
    /// (`0` = uncapped). Early-stopping range over `(cursor, ..]`: O(result +
    /// log n), never an O(n) scan, and nothing in the window is missed or
    /// duplicated.
    pub fn read_after(&self, cursor: u64, limit: usize) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        for (_token, event) in self
            .by_token
            .range((Bound::Excluded(cursor), Bound::Unbounded))
        {
            out.push(event.clone());
            if limit != 0 && out.len() >= limit {
                break;
            }
        }
        out
    }

    /// The highest token currently in the log, or `0` if empty.
    pub fn latest_token(&self) -> u64 {
        self.by_token
            .get_max()
            .map(|(token, _)| *token)
            .unwrap_or(0)
    }

    /// The token the next [`append`](Self::append) will assign.
    pub fn next_token(&self) -> u64 {
        self.next_token.max(1)
    }

    pub fn len(&self) -> usize {
        self.by_token.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_token.is_empty()
    }
}

/// The delta a [`read`](StreamRegistry::read) returns for one stream: the events
/// plus the cursor callers should persist when `advance=true`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamDelta {
    pub stream_key: String,
    pub events: Vec<StreamEvent>,
    pub new_cursor: u64,
}

/// Local embedded live-tail registry: many [`StreamLog`]s plus per-`(actor,
/// stream)` cursors and per-actor subscription sets.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StreamRegistry {
    streams: BTreeMap<String, StreamLog>,
    cursors: BTreeMap<(String, String), u64>,
    subscriptions: BTreeMap<String, BTreeSet<String>>,
}

impl StreamRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn append(&mut self, stream_key: &str, event: StreamEvent) -> u64 {
        self.streams
            .entry(stream_key.to_string())
            .or_default()
            .append(event)
    }

    pub fn cursor(&self, actor: &str, stream_key: &str) -> u64 {
        self.cursors
            .get(&(actor.to_string(), stream_key.to_string()))
            .copied()
            .unwrap_or(0)
    }

    pub fn read(
        &mut self,
        actor: &str,
        stream_key: &str,
        advance: bool,
        limit: usize,
    ) -> StreamDelta {
        let cursor = self.cursor(actor, stream_key);
        let events = self
            .streams
            .get(stream_key)
            .map(|log| log.read_after(cursor, limit))
            .unwrap_or_default();
        let new_cursor = events
            .last()
            .map(|event| event.ordering_token)
            .unwrap_or(cursor);
        if advance && new_cursor != cursor {
            self.cursors
                .insert((actor.to_string(), stream_key.to_string()), new_cursor);
        }
        StreamDelta {
            stream_key: stream_key.to_string(),
            events,
            new_cursor,
        }
    }

    pub fn subscribe(&mut self, actor: &str, stream_key: &str) -> Vec<String> {
        self.subscriptions
            .entry(actor.to_string())
            .or_default()
            .insert(stream_key.to_string());
        self.subscriptions(actor)
    }

    pub fn unsubscribe(&mut self, actor: &str, stream_key: &str) -> Vec<String> {
        if let Some(set) = self.subscriptions.get_mut(actor) {
            set.remove(stream_key);
            if set.is_empty() {
                self.subscriptions.remove(actor);
            }
        }
        self.subscriptions(actor)
    }

    pub fn subscriptions(&self, actor: &str) -> Vec<String> {
        self.subscriptions
            .get(actor)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }

    pub fn read_subscribed(
        &mut self,
        actor: &str,
        advance: bool,
        limit: usize,
    ) -> Vec<StreamDelta> {
        let streams = self.subscriptions(actor);
        streams
            .into_iter()
            .map(|stream_key| self.read(actor, &stream_key, advance, limit))
            .filter(|delta| !delta.events.is_empty())
            .collect()
    }

    pub fn latest_token(&self, stream_key: &str) -> u64 {
        self.streams
            .get(stream_key)
            .map(StreamLog::latest_token)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn store() -> StreamStore {
        StreamStore::default()
    }

    fn tokens(events: &[StreamEvent]) -> Vec<u64> {
        events.iter().map(|e| e.ordering_token).collect()
    }

    fn log_event(actor: &str, kind: &str) -> StreamEvent {
        StreamEvent::new(
            format!("evt:{actor}:{kind}"),
            "tenant\u{1}room:demo",
            actor,
            kind,
            json!({ "k": kind }),
            StreamUrgency::Info,
            None,
            "t",
        )
    }

    #[test]
    fn append_assigns_strictly_increasing_tokens() {
        let mut log = StreamLog::new();
        let a = log.append(log_event("codex", "a"));
        let b = log.append(log_event("claude-code", "b"));
        let c = log.append(log_event("codex", "c"));
        assert_eq!((a, b, c), (1, 2, 3));
        assert_eq!(log.latest_token(), 3);
        assert_eq!(log.next_token(), 4);
    }

    #[test]
    fn read_after_returns_exact_ordered_delta_with_no_miss_or_dup() {
        let mut log = StreamLog::new();
        for kind in ["a", "b", "c", "d", "e"] {
            log.append(log_event("codex", kind));
        }
        assert_eq!(tokens(&log.read_after(0, 0)), vec![1, 2, 3, 4, 5]);
        assert_eq!(tokens(&log.read_after(2, 0)), vec![3, 4, 5]);
        assert!(log.read_after(5, 0).is_empty());
        assert_eq!(tokens(&log.read_after(0, 2)), vec![1, 2]);
    }

    #[test]
    fn stream_log_concurrent_publishers_get_distinct_tokens_in_one_total_order() {
        let mut log = StreamLog::new();
        let t_codex = log.append(log_event("codex", "x"));
        let t_claude = log.append(log_event("claude-code", "y"));
        assert_ne!(t_codex, t_claude);
        let order = log
            .read_after(0, 0)
            .into_iter()
            .map(|event| (event.actor, event.ordering_token))
            .collect::<Vec<_>>();
        assert_eq!(
            order,
            vec![("codex".to_string(), 1), ("claude-code".to_string(), 2)]
        );
    }

    #[test]
    fn ingest_preserves_tokens_and_keeps_append_monotonic() {
        let mut log = StreamLog::new();
        let mut e7 = log_event("codex", "seven");
        e7.ordering_token = 7;
        let mut e3 = log_event("codex", "three");
        e3.ordering_token = 3;
        log.ingest(e7);
        log.ingest(e3);
        assert_eq!(tokens(&log.read_after(0, 0)), vec![3, 7]);
        assert_eq!(log.append(log_event("codex", "next")), 8);
    }

    #[test]
    fn registry_cursor_advances_and_consumes_window_once() {
        let mut reg = StreamRegistry::new();
        let key = "tenant\u{1}room:demo";
        reg.append(key, log_event("codex", "a"));
        reg.append(key, log_event("codex", "b"));

        let first = reg.read("claude-code", key, true, 0);
        assert_eq!(first.events.len(), 2);
        assert_eq!(first.new_cursor, 2);
        let second = reg.read("claude-code", key, true, 0);
        assert!(second.events.is_empty());
        assert_eq!(second.new_cursor, 2);
        let other = reg.read("deepseek", key, true, 0);
        assert_eq!(other.events.len(), 2);
    }

    #[test]
    fn passive_read_without_advance_is_repeatable() {
        let mut reg = StreamRegistry::new();
        let key = "tenant\u{1}room:demo";
        reg.append(key, log_event("codex", "a"));
        let peek_a = reg.read("claude-code", key, false, 0);
        let peek_b = reg.read("claude-code", key, false, 0);
        assert_eq!(peek_a.events.len(), 1);
        assert_eq!(peek_b.events.len(), 1);
        assert_eq!(reg.cursor("claude-code", key), 0);
    }

    #[test]
    fn subscription_set_controls_which_deltas_a_read_returns() {
        let mut reg = StreamRegistry::new();
        let room = "tenant\u{1}room:demo";
        let task = "tenant\u{1}task:stream";
        reg.append(room, log_event("codex", "room-1"));
        reg.append(task, log_event("codex", "task-1"));

        assert!(reg.read_subscribed("claude-code", true, 0).is_empty());

        let set = reg.subscribe("claude-code", room);
        assert_eq!(set, vec![room.to_string()]);
        let deltas = reg.read_subscribed("claude-code", true, 0);
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].stream_key, room);

        reg.subscribe("claude-code", task);
        reg.append(room, log_event("codex", "room-2"));
        let both = reg.read_subscribed("claude-code", true, 0);
        let keys = both
            .iter()
            .map(|delta| delta.stream_key.clone())
            .collect::<BTreeSet<_>>();
        assert!(keys.contains(room));
        assert!(keys.contains(task));

        let remaining = reg.unsubscribe("claude-code", task);
        assert_eq!(remaining, vec![room.to_string()]);
        reg.append(task, log_event("codex", "task-2"));
        let after = reg.read_subscribed("claude-code", true, 0);
        assert!(after.iter().all(|delta| delta.stream_key != task));
    }

    #[test]
    fn stream_key_canonical_is_separator_safe() {
        let key = StreamKey::new("travis-gilbert", "room:demo");
        assert_eq!(key.canonical(), "travis-gilbert\u{1}room:demo");
        let a = StreamKey::new("a", "b").canonical();
        let b = StreamKey::new("a\u{1}b", "").canonical();
        assert_ne!(a, b);
    }

    // --- Acceptance criterion 1 -------------------------------------------
    // A head offline for N turns, on reconnect, pulls exactly the events on its
    // subscribed streams after its stored cursor, in order, cursor advancing;
    // nothing in that window is missed or duplicated.
    #[test]
    fn offline_head_pulls_exact_delta_after_cursor() {
        let mut s = store();
        let key = StreamStore::resolve_stream_key("acme", "room").unwrap();
        s.subscribe("acme", "alice", &key).unwrap();

        for i in 0..3 {
            s.publish(
                "acme",
                "room",
                "bob",
                "msg",
                json!({ "i": i }),
                StreamUrgency::Info,
                None,
            )
            .unwrap();
        }
        let (first, cursors) = s.read("acme", "alice", &[], true, 0).unwrap();
        assert_eq!(tokens(&first), vec![1, 2, 3]);
        assert_eq!(cursors[&key], 3);

        // Offline for N turns; two more events land.
        s.publish(
            "acme",
            "room",
            "bob",
            "msg",
            json!({ "i": 3 }),
            StreamUrgency::Info,
            None,
        )
        .unwrap();
        s.publish(
            "acme",
            "room",
            "bob",
            "msg",
            json!({ "i": 4 }),
            StreamUrgency::Info,
            None,
        )
        .unwrap();

        // On reconnect, EXACTLY the two events after the cursor, in order.
        let (delta, cursors) = s.read("acme", "alice", &[], true, 0).unwrap();
        assert_eq!(tokens(&delta), vec![4, 5]);
        assert_eq!(cursors[&key], 5);

        // Idempotent re-read returns nothing -- no duplicates.
        let (empty, _) = s.read("acme", "alice", &[], true, 0).unwrap();
        assert!(empty.is_empty());
    }

    // --- HANDOFF-HEAD-CALLS D2: JetStream-grade ack semantics -------------

    #[test]
    fn explicit_ack_redelivers_unacked_after_crash() {
        let mut s = store();
        let key = StreamStore::resolve_stream_key("acme", "room").unwrap();
        s.subscribe("acme", "alice", &key).unwrap();
        s.publish(
            "acme",
            "room",
            "bob",
            "head_call",
            json!({ "n": 1 }),
            StreamUrgency::Info,
            None,
        )
        .unwrap();

        // First delivery under explicit ack: delivered, pending count 1.
        let first = s
            .read_delivery(
                "acme",
                "alice",
                std::slice::from_ref(&key),
                AckPolicy::Explicit,
                1000,
                0,
                DEFAULT_MAX_DELIVER,
                0,
            )
            .unwrap();
        assert_eq!(tokens(&first.events), vec![1]);
        assert_eq!(first.pending.len(), 1);
        assert_eq!(first.pending[0].delivered_count, 1);
        assert!(first.dead_lettered.is_empty());

        // Consumer crashes before acking; a fresh read redelivers with count 2.
        let second = s
            .read_delivery(
                "acme",
                "alice",
                std::slice::from_ref(&key),
                AckPolicy::Explicit,
                1000,
                0,
                DEFAULT_MAX_DELIVER,
                0,
            )
            .unwrap();
        assert_eq!(tokens(&second.events), vec![1]);
        assert_eq!(second.pending[0].delivered_count, 2);
    }

    #[test]
    fn explicit_ack_eviction_is_reported_as_dead_letter() {
        let mut s = store();
        let key = StreamStore::resolve_stream_key("acme", "room").unwrap();
        s.subscribe("acme", "alice", &key).unwrap();
        s.publish(
            "acme",
            "room",
            "bob",
            "head_call",
            json!({ "n": 1 }),
            StreamUrgency::Info,
            None,
        )
        .unwrap();
        let first = s
            .read_delivery(
                "acme",
                "alice",
                std::slice::from_ref(&key),
                AckPolicy::Explicit,
                1000,
                0,
                DEFAULT_MAX_DELIVER,
                0,
            )
            .unwrap();
        assert_eq!(tokens(&first.events), vec![1]);

        for n in 0..MAX_STREAM_LOG_LEN {
            s.publish(
                "acme",
                "room",
                "bob",
                "filler",
                json!({ "n": n }),
                StreamUrgency::Info,
                None,
            )
            .unwrap();
        }
        let after_eviction = s
            .read_delivery(
                "acme",
                "alice",
                std::slice::from_ref(&key),
                AckPolicy::Explicit,
                2000,
                0,
                DEFAULT_MAX_DELIVER,
                0,
            )
            .unwrap();

        assert_eq!(after_eviction.dead_lettered.len(), 1);
        assert_eq!(after_eviction.dead_lettered[0].ordering_token, 1);
        assert_eq!(after_eviction.dead_lettered[0].kind, "evicted_before_ack");
        assert_eq!(
            after_eviction.dead_lettered[0].target_actor.as_deref(),
            Some("alice")
        );
    }

    #[test]
    fn ack_stops_redelivery() {
        let mut s = store();
        let key = StreamStore::resolve_stream_key("acme", "room").unwrap();
        s.subscribe("acme", "alice", &key).unwrap();
        s.publish(
            "acme",
            "room",
            "bob",
            "head_call",
            json!({}),
            StreamUrgency::Info,
            None,
        )
        .unwrap();

        let first = s
            .read_delivery(
                "acme",
                "alice",
                std::slice::from_ref(&key),
                AckPolicy::Explicit,
                1000,
                0,
                DEFAULT_MAX_DELIVER,
                0,
            )
            .unwrap();
        let token = first.events[0].ordering_token;
        assert_eq!(s.ack("acme", "alice", &key, &[token]).unwrap(), 1);

        // Acked: no redelivery, no new events.
        let after = s
            .read_delivery(
                "acme",
                "alice",
                std::slice::from_ref(&key),
                AckPolicy::Explicit,
                5000,
                0,
                DEFAULT_MAX_DELIVER,
                0,
            )
            .unwrap();
        assert!(after.events.is_empty());
        assert!(after.dead_lettered.is_empty());
    }

    #[test]
    fn explicit_max_deliver_dead_letters_and_stops() {
        let mut s = store();
        let key = StreamStore::resolve_stream_key("acme", "room").unwrap();
        s.subscribe("acme", "alice", &key).unwrap();
        s.publish(
            "acme",
            "room",
            "bob",
            "head_call",
            json!({}),
            StreamUrgency::Info,
            None,
        )
        .unwrap();

        let max = 3u32;
        // Deliveries 1..=max, never acked (redeliver_after 0 so each read redelivers).
        for expect in 1..=max {
            let r = s
                .read_delivery(
                    "acme",
                    "alice",
                    std::slice::from_ref(&key),
                    AckPolicy::Explicit,
                    1000,
                    0,
                    max,
                    0,
                )
                .unwrap();
            assert_eq!(r.pending.len(), 1);
            assert_eq!(r.pending[0].delivered_count, expect);
            assert!(r.dead_lettered.is_empty());
        }
        // Next read: the count is at max, so the event dead-letters and drops.
        let dead = s
            .read_delivery(
                "acme",
                "alice",
                std::slice::from_ref(&key),
                AckPolicy::Explicit,
                1000,
                0,
                max,
                0,
            )
            .unwrap();
        assert_eq!(tokens(&dead.dead_lettered), vec![1]);
        assert!(dead.events.is_empty());
        // Redelivery has stopped.
        let after = s
            .read_delivery(
                "acme",
                "alice",
                std::slice::from_ref(&key),
                AckPolicy::Explicit,
                1000,
                0,
                max,
                0,
            )
            .unwrap();
        assert!(after.events.is_empty());
        assert!(after.dead_lettered.is_empty());
    }

    #[test]
    fn on_read_policy_matches_classic_read() {
        // OnRead delivery must be byte-identical to today's read(advance=true).
        let mut a = store();
        let mut b = store();
        let key = StreamStore::resolve_stream_key("acme", "room").unwrap();
        a.subscribe("acme", "alice", &key).unwrap();
        b.subscribe("acme", "alice", &key).unwrap();
        for i in 0..2 {
            a.publish(
                "acme",
                "room",
                "bob",
                "msg",
                json!({ "i": i }),
                StreamUrgency::Info,
                None,
            )
            .unwrap();
            b.publish(
                "acme",
                "room",
                "bob",
                "msg",
                json!({ "i": i }),
                StreamUrgency::Info,
                None,
            )
            .unwrap();
        }

        let classic = a.read("acme", "alice", &[], true, 0).unwrap();
        let delivery = b
            .read_delivery(
                "acme",
                "alice",
                &[],
                AckPolicy::OnRead,
                1000,
                DEFAULT_REDELIVER_AFTER_MS,
                DEFAULT_MAX_DELIVER,
                0,
            )
            .unwrap();
        assert_eq!(tokens(&classic.0), tokens(&delivery.events));
        assert_eq!(classic.1, delivery.cursors);
        assert!(delivery.pending.is_empty());

        // Cursor advanced identically: both re-read empty.
        assert!(a.read("acme", "alice", &[], true, 0).unwrap().0.is_empty());
        assert!(b
            .read_delivery(
                "acme",
                "alice",
                &[],
                AckPolicy::OnRead,
                1000,
                0,
                DEFAULT_MAX_DELIVER,
                0
            )
            .unwrap()
            .events
            .is_empty());
    }

    #[test]
    fn explicit_ack_respects_global_limit_across_streams() {
        let mut s = store();
        let alpha = StreamStore::resolve_stream_key("acme", "alpha").unwrap();
        let beta = StreamStore::resolve_stream_key("acme", "beta").unwrap();
        s.subscribe("acme", "alice", &alpha).unwrap();
        s.subscribe("acme", "alice", &beta).unwrap();

        s.publish(
            "acme",
            "alpha",
            "bob",
            "msg",
            json!({ "n": 1 }),
            StreamUrgency::Info,
            None,
        )
        .unwrap();
        s.publish(
            "acme",
            "beta",
            "bob",
            "msg",
            json!({ "n": 2 }),
            StreamUrgency::Info,
            None,
        )
        .unwrap();
        s.publish(
            "acme",
            "alpha",
            "bob",
            "msg",
            json!({ "n": 3 }),
            StreamUrgency::Info,
            None,
        )
        .unwrap();

        let first = s
            .read_delivery(
                "acme",
                "alice",
                &[],
                AckPolicy::Explicit,
                1000,
                DEFAULT_REDELIVER_AFTER_MS,
                DEFAULT_MAX_DELIVER,
                2,
            )
            .unwrap();
        assert_eq!(tokens(&first.events), vec![1, 2]);
        assert_eq!(first.pending.len(), 2);
        assert_eq!(first.cursors[&alpha], 1);
        assert_eq!(first.cursors[&beta], 2);

        let second = s
            .read_delivery(
                "acme",
                "alice",
                &[],
                AckPolicy::Explicit,
                1001,
                DEFAULT_REDELIVER_AFTER_MS,
                DEFAULT_MAX_DELIVER,
                2,
            )
            .unwrap();
        assert_eq!(tokens(&second.events), vec![3]);
        assert_eq!(second.cursors[&alpha], 3);
        assert_eq!(second.cursors[&beta], 2);
    }

    // --- Acceptance criterion 2 -------------------------------------------
    // A ping (urgency ask|block + target) to a warm head appears in its next
    // mention drain; the same queue triggers a cold-head wake.
    #[test]
    fn ping_lands_in_targets_mention_drain() {
        let mut s = store();
        let event = s
            .publish(
                "acme",
                "room",
                "alice",
                "question",
                json!({ "q": "review?" }),
                StreamUrgency::Ask,
                Some("bob".to_string()),
            )
            .unwrap();

        let mentions = s.drain_mentions("acme", "bob", true).unwrap();
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].id, event.id);
        assert_eq!(mentions[0].urgency, StreamUrgency::Ask);

        // Delivery is retryable until explicitly acknowledged.
        assert_eq!(s.drain_mentions("acme", "bob", true).unwrap().len(), 1);
        assert_eq!(
            s.ack_mentions("acme", "bob", &[event.ordering_token])
                .unwrap(),
            1
        );
        assert!(s.drain_mentions("acme", "bob", true).unwrap().is_empty());

        // A `block` ping queues just the same; peek does not drain.
        let blocked = s
            .publish(
                "acme",
                "room",
                "alice",
                "blocker",
                json!({}),
                StreamUrgency::Block,
                Some("bob".to_string()),
            )
            .unwrap();
        assert_eq!(s.drain_mentions("acme", "bob", false).unwrap().len(), 1);
        assert_eq!(s.drain_mentions("acme", "bob", true).unwrap().len(), 1);
        s.ack_mentions("acme", "bob", &[blocked.ordering_token])
            .unwrap();
        assert!(s.drain_mentions("acme", "bob", true).unwrap().is_empty());
    }

    // --- Acceptance criterion 3 -------------------------------------------
    // Two heads publishing concurrently to one stream receive distinct ordering
    // tokens; both events are readable in a single total order with no merge.
    #[test]
    fn concurrent_publishers_get_distinct_tokens_in_total_order() {
        let mut s = store();
        let e1 = s
            .publish(
                "acme",
                "room",
                "alice",
                "msg",
                json!({}),
                StreamUrgency::Info,
                None,
            )
            .unwrap();
        let e2 = s
            .publish(
                "acme",
                "room",
                "bob",
                "msg",
                json!({}),
                StreamUrgency::Info,
                None,
            )
            .unwrap();
        assert_ne!(e1.ordering_token, e2.ordering_token);
        assert!(e1.ordering_token < e2.ordering_token);

        let key = StreamStore::resolve_stream_key("acme", "room").unwrap();
        let (events, _) = s.read("acme", "carol", &[key], false, 0).unwrap();
        assert_eq!(tokens(&events), vec![e1.ordering_token, e2.ordering_token]);
    }

    // --- Acceptance criterion 4 -------------------------------------------
    // A publish and a read under the configured tenant share a stream; a call
    // with an empty tenant is rejected, not silently routed to a default.
    #[test]
    fn tenant_scope_shares_stream_and_rejects_empty() {
        let mut s = store();
        s.publish(
            "acme",
            "room",
            "alice",
            "msg",
            json!({}),
            StreamUrgency::Info,
            None,
        )
        .unwrap();
        let key = StreamStore::resolve_stream_key("acme", "room").unwrap();
        let (events, _) = s.read("acme", "bob", &[key], false, 0).unwrap();
        assert_eq!(events.len(), 1);

        // Empty / whitespace-only tenant is rejected, never defaulted.
        assert_eq!(
            s.publish("", "room", "a", "msg", json!({}), StreamUrgency::Info, None)
                .unwrap_err()
                .code,
            "empty_tenant"
        );
        assert_eq!(
            StreamStore::resolve_stream_key("   ", "room")
                .unwrap_err()
                .code,
            "empty_tenant"
        );
        // An empty topic is rejected too.
        assert_eq!(
            StreamStore::resolve_stream_key("acme", "  ")
                .unwrap_err()
                .code,
            "empty_stream_topic"
        );
        // A non-empty tenant always resolves (the normalizer is total).
        assert!(StreamStore::resolve_stream_key("acme", "room").is_ok());
        let beta_key = StreamStore::resolve_stream_key("beta", "room").unwrap();
        assert_eq!(
            s.read("acme", "bob", std::slice::from_ref(&beta_key), false, 0)
                .unwrap_err()
                .code,
            "tenant_mismatch"
        );
        assert_eq!(
            s.subscribe("acme", "bob", &beta_key).unwrap_err().code,
            "tenant_mismatch"
        );
    }

    // --- Acceptance criterion 5 -------------------------------------------
    // Subscribing and unsubscribing changes which streams' deltas a read
    // returns; a ping still reaches an unsubscribed target.
    #[test]
    fn attention_controls_reads_but_ping_bypasses_it() {
        let mut s = store();
        let key = StreamStore::resolve_stream_key("acme", "room").unwrap();

        s.publish(
            "acme",
            "room",
            "alice",
            "msg",
            json!({ "n": 0 }),
            StreamUrgency::Info,
            None,
        )
        .unwrap();
        let (none, _) = s.read("acme", "bob", &[], true, 0).unwrap();
        assert!(none.is_empty(), "no subscription => empty delta");

        s.subscribe("acme", "bob", &key).unwrap();
        s.publish(
            "acme",
            "room",
            "alice",
            "msg",
            json!({ "n": 1 }),
            StreamUrgency::Info,
            None,
        )
        .unwrap();
        let (after_sub, _) = s.read("acme", "bob", &[], true, 0).unwrap();
        assert_eq!(after_sub.len(), 1);

        s.unsubscribe("acme", "bob", &key).unwrap();
        s.publish(
            "acme",
            "room",
            "alice",
            "msg",
            json!({ "n": 2 }),
            StreamUrgency::Info,
            None,
        )
        .unwrap();
        let (after_unsub, _) = s.read("acme", "bob", &[], true, 0).unwrap();
        assert!(after_unsub.is_empty(), "unsubscribed => no delta");

        // A ping still reaches bob even though he is not subscribed.
        s.publish(
            "acme",
            "room",
            "alice",
            "ping",
            json!({}),
            StreamUrgency::Block,
            Some("bob".to_string()),
        )
        .unwrap();
        assert_eq!(
            s.drain_mentions("acme", "bob", true).unwrap().len(),
            1,
            "ping bypasses attention"
        );
    }

    #[test]
    fn urgency_parses_and_rejects_unknown() {
        assert_eq!(StreamUrgency::parse(""), Some(StreamUrgency::Info));
        assert_eq!(StreamUrgency::parse("INFO"), Some(StreamUrgency::Info));
        assert_eq!(StreamUrgency::parse(" ask "), Some(StreamUrgency::Ask));
        assert_eq!(StreamUrgency::parse("block"), Some(StreamUrgency::Block));
        assert_eq!(StreamUrgency::parse("urgent"), None);
    }

    #[test]
    fn read_limit_caps_the_window_and_resumes() {
        let mut s = store();
        let key = StreamStore::resolve_stream_key("acme", "room").unwrap();
        for i in 0..10 {
            s.publish(
                "acme",
                "room",
                "a",
                "msg",
                json!({ "i": i }),
                StreamUrgency::Info,
                None,
            )
            .unwrap();
        }
        let (events, cursors) = s
            .read("acme", "alice", std::slice::from_ref(&key), true, 4)
            .unwrap();
        assert_eq!(tokens(&events), vec![1, 2, 3, 4]);
        assert_eq!(cursors[&key], 4);
        let (next, _) = s.read("acme", "alice", &[key], true, 4).unwrap();
        assert_eq!(tokens(&next), vec![5, 6, 7, 8]);
    }

    #[test]
    fn tenant_scope_isolates_subscription_reads_and_mentions() {
        let mut s = store();
        let acme_room = StreamStore::resolve_stream_key("acme", "room").unwrap();
        let beta_room = StreamStore::resolve_stream_key("beta", "room").unwrap();
        s.subscribe("acme", "codex", &acme_room).unwrap();
        s.subscribe("beta", "codex", &beta_room).unwrap();

        let acme_event = s
            .publish(
                "acme",
                "room",
                "alice",
                "msg",
                json!({ "tenant": "acme" }),
                StreamUrgency::Info,
                None,
            )
            .unwrap();
        let beta_event = s
            .publish(
                "beta",
                "room",
                "alice",
                "msg",
                json!({ "tenant": "beta" }),
                StreamUrgency::Info,
                None,
            )
            .unwrap();

        let (acme_events, acme_cursors) = s.read("acme", "codex", &[], true, 0).unwrap();
        assert_eq!(acme_events.len(), 1);
        assert_eq!(acme_events[0].id, acme_event.id);
        assert_eq!(acme_events[0].stream_key, acme_room);
        assert_eq!(acme_cursors.get(&beta_room), None);

        let (beta_events, beta_cursors) = s.read("beta", "codex", &[], true, 0).unwrap();
        assert_eq!(beta_events.len(), 1);
        assert_eq!(beta_events[0].id, beta_event.id);
        assert_eq!(beta_events[0].stream_key, beta_room);
        assert_eq!(beta_cursors.get(&acme_room), None);

        let acme_ping = s
            .publish(
                "acme",
                "room",
                "alice",
                "ask",
                json!({}),
                StreamUrgency::Ask,
                Some(" codex ".to_string()),
            )
            .unwrap();
        assert_eq!(acme_ping.target_actor.as_deref(), Some("codex"));
        let beta_ping = s
            .publish(
                "beta",
                "room",
                "alice",
                "ask",
                json!({}),
                StreamUrgency::Ask,
                Some("codex".to_string()),
            )
            .unwrap();

        let acme_mentions = s.drain_mentions("acme", "codex", true).unwrap();
        assert_eq!(acme_mentions.len(), 1);
        assert_eq!(acme_mentions[0].id, acme_ping.id);
        s.ack_mentions("acme", "codex", &[acme_ping.ordering_token])
            .unwrap();
        assert!(s.drain_mentions("acme", "codex", true).unwrap().is_empty());

        let beta_mentions = s.drain_mentions("beta", "codex", true).unwrap();
        assert_eq!(beta_mentions.len(), 1);
        assert_eq!(beta_mentions[0].id, beta_ping.id);
    }

    #[test]
    fn multi_stream_limit_is_global_and_cursors_match_returned_events() {
        let mut s = store();
        let alpha = StreamStore::resolve_stream_key("acme", "alpha").unwrap();
        let beta = StreamStore::resolve_stream_key("acme", "beta").unwrap();
        s.subscribe("acme", "reader", &alpha).unwrap();
        s.subscribe("acme", "reader", &beta).unwrap();

        s.publish(
            "acme",
            "alpha",
            "a",
            "msg",
            json!({}),
            StreamUrgency::Info,
            None,
        )
        .unwrap();
        s.publish(
            "acme",
            "alpha",
            "a",
            "msg",
            json!({}),
            StreamUrgency::Info,
            None,
        )
        .unwrap();
        s.publish(
            "acme",
            "beta",
            "b",
            "msg",
            json!({}),
            StreamUrgency::Info,
            None,
        )
        .unwrap();
        s.publish(
            "acme",
            "beta",
            "b",
            "msg",
            json!({}),
            StreamUrgency::Info,
            None,
        )
        .unwrap();

        let (first, cursors) = s.read("acme", "reader", &[], true, 2).unwrap();
        assert_eq!(tokens(&first), vec![1, 2]);
        assert_eq!(cursors[&alpha], 2);
        assert_eq!(cursors[&beta], 0);

        let (second, cursors) = s.read("acme", "reader", &[], true, 2).unwrap();
        assert_eq!(tokens(&second), vec![3, 4]);
        assert_eq!(cursors[&alpha], 2);
        assert_eq!(cursors[&beta], 4);
    }

    #[test]
    fn stream_logs_and_pending_ping_queues_are_bounded() {
        let mut s = store();
        let key = StreamStore::resolve_stream_key("acme", "room").unwrap();
        for i in 0..(MAX_STREAM_LOG_LEN + 5) {
            s.publish(
                "acme",
                "room",
                "a",
                "msg",
                json!({ "i": i }),
                StreamUrgency::Info,
                None,
            )
            .unwrap();
        }
        let (events, _) = s
            .read("acme", "reader", &[key], false, MAX_STREAM_LOG_LEN + 10)
            .unwrap();
        assert_eq!(events.len(), MAX_STREAM_LOG_LEN);
        assert_eq!(events.first().map(|event| event.ordering_token), Some(6));

        for i in 0..(MAX_PENDING_PINGS_PER_ACTOR + 5) {
            s.publish(
                "acme",
                "room",
                "a",
                "ask",
                json!({ "i": i }),
                StreamUrgency::Ask,
                Some("target".to_string()),
            )
            .unwrap();
        }
        let mentions = s.drain_mentions("acme", "target", false).unwrap();
        assert_eq!(mentions.len(), MAX_PENDING_PINGS_PER_ACTOR);
        assert_eq!(
            mentions.first().map(|event| event.ordering_token),
            Some((MAX_STREAM_LOG_LEN + 11) as u64)
        );
    }

    #[test]
    fn actors_and_ping_targets_are_validated() {
        let mut s = store();
        assert_eq!(
            s.publish(
                "acme",
                "room",
                "",
                "msg",
                json!({}),
                StreamUrgency::Info,
                None
            )
            .unwrap_err()
            .code,
            "empty_actor"
        );
        assert_eq!(
            s.publish(
                "acme",
                "room",
                "alice|eve",
                "msg",
                json!({}),
                StreamUrgency::Info,
                None
            )
            .unwrap_err()
            .code,
            "invalid_actor"
        );
        assert_eq!(
            s.publish(
                "acme",
                "room",
                "alice",
                "ask",
                json!({}),
                StreamUrgency::Ask,
                Some("  ".to_string())
            )
            .unwrap_err()
            .code,
            "missing_target_actor"
        );
    }
}
