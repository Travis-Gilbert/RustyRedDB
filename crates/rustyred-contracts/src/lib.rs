//! Shared database contracts.
//!
//! These types used to live beside UI and shell records. The database keeps
//! the ones it serializes: sync cursors, file ownership, article history, and
//! link records. Serialized names are unchanged.

pub mod article;
pub mod article_links;
pub mod desktop;
pub mod link;
pub mod link_api;
pub mod sync;
pub mod wiki;
