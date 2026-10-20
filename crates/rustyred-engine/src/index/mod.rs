//! Access-method index implementations that are large enough to own a module.
//!
//! The flat modules beside this one (`ordered`, `fulltext`, `spatial`) predate
//! this directory and stay where they are. New multi-type index families land
//! here so the crate root does not keep growing sideways.

pub mod trigram;

pub use trigram::{
    CandidateSet, TrigramAccessMethod, TrigramIndex, TrigramIndexDefinition, TRIGRAM_ACCESS_METHOD,
};
