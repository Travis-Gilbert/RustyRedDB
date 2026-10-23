use std::path::{Path, PathBuf};
use std::sync::Arc;

use rustyred_thg_core::file_payload::{
    install_file_manifest_reference, read_file_payload, stage_file_payload, FileManifestRef,
    FILE_CONTENT_MANIFEST_PROPERTY,
};
#[cfg(feature = "fulltext")]
use rustyred_thg_core::fulltext_store::FullTextHit;
use rustyred_thg_core::reachability::{compact_committed_reachable, CoreReachabilityCodec};
use rustyred_thg_core::rr_file::RustyRed;
use rustyred_thg_core::rr_migrate::migrate_journal_to_rr;
use rustyred_thg_core::{
    CompactionReport, GraphMutationBatch, GraphStats, GraphTransaction, NodeRecord,
    RedCoreDurability, RedCoreGraphStore, RedCoreOptions, UnifiedChunkStore, VerifyReport,
    REDCORE_JOURNAL_FILE,
};
use serde::Serialize;
use theorem_surface_contracts::desktop::DeviceId;

use crate::error::{Error, ErrorKind};
use crate::options::OpenOptions;

/// One open Rusty Red database.
///
/// The native file handle, when there is one, is dropped after the graph so
/// the writer claim stays held for the whole life of this value and every
/// chunk-store clone it handed out.
pub struct Database {
    graph: RedCoreGraphStore,
    file: Option<RustyRed>,
    writable: bool,
    path: Option<PathBuf>,
}

/// Counts and identity for `inspect`.
#[derive(Clone, Debug, Serialize)]
pub struct Inspection {
    pub path: Option<String>,
    pub database_id: Option<String>,
    pub writable: bool,
    pub version: u64,
    pub nodes: usize,
    pub edges: usize,
}

/// Result of copying a directory journal into a new `.RR` file.
///
/// The source journal is left byte-for-byte as it was. The target is a new
/// file with its own database id.
#[derive(Clone, Debug)]
pub struct MigrationReport {
    pub source_preserved: bool,
    pub chunks_copied: usize,
    pub named_roots: usize,
    pub database_id: String,
}

impl Database {
    /// Create a new `.RR` file. Refuses when `path` already exists.
    pub fn create(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::create_with(path, &OpenOptions::default())
    }

    pub fn create_with(path: impl AsRef<Path>, options: &OpenOptions) -> Result<Self, Error> {
        let path = path.as_ref();
        if path.exists() {
            return Err(Error::new(
                ErrorKind::AlreadyExists,
                "rr_already_exists",
                format!(
                    "{} already exists; open it instead of creating over it",
                    path.display()
                ),
            ));
        }
        let device = DeviceId::new(&options.device_name);
        let file = RustyRed::create(path, &device).map_err(Error::from_storage)?;
        let graph = open_graph(file.store().clone(), options)?;
        Ok(Self {
            graph,
            file: Some(file),
            writable: true,
            path: Some(path.to_path_buf()),
        })
    }

    /// Open an existing `.RR` file for writing. A missing path is [`ErrorKind::NotFound`]
    /// and is not created. A second writer is refused while this value, or any
    /// chunk-store clone taken from it, is still alive.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::open_with(path, &OpenOptions::default())
    }

    pub fn open_with(path: impl AsRef<Path>, options: &OpenOptions) -> Result<Self, Error> {
        let path = path.as_ref();
        require_existing(path)?;
        let device = DeviceId::new(&options.device_name);
        let file = RustyRed::open(path, &device).map_err(Error::from_storage)?;
        let mut graph = open_graph(file.store().clone(), options)?;
        graph
            .publish_rebuilt_fulltext()
            .map_err(Error::from_storage)?;
        Ok(Self {
            graph,
            file: Some(file),
            writable: true,
            path: Some(path.to_path_buf()),
        })
    }

    /// Open an existing `.RR` file without a writer claim.
    ///
    /// Reads are the catalog recovered at open. This handle does not refresh
    /// itself when another process commits, and it does not repair, compact,
    /// or publish a root.
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        require_existing(path)?;
        let file = RustyRed::open_read_only(path).map_err(Error::from_storage)?;
        let graph = open_graph(file.store().clone(), &OpenOptions::default())?;
        Ok(Self {
            graph,
            file: Some(file),
            writable: false,
            path: Some(path.to_path_buf()),
        })
    }

    /// An empty graph with no files and no durability.
    pub fn in_memory() -> Self {
        Self {
            graph: RedCoreGraphStore::memory(),
            file: None,
            writable: true,
            path: None,
        }
    }

    /// Create a directory-backed database, the deployment shape that stores
    /// `redcore.journal` beside its payloads. Refuses when that journal is
    /// already present. This is not the single-file format.
    pub fn create_directory(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::create_directory_with(path, &OpenOptions::default())
    }

    pub fn create_directory_with(
        path: impl AsRef<Path>,
        options: &OpenOptions,
    ) -> Result<Self, Error> {
        let path = path.as_ref();
        let journal = path.join(REDCORE_JOURNAL_FILE);
        if journal.exists() {
            return Err(Error::new(
                ErrorKind::AlreadyExists,
                "rr_already_exists",
                format!("{} already exists", journal.display()),
            ));
        }
        let mut graph =
            RedCoreGraphStore::open(path, directory_options()).map_err(Error::from_storage)?;
        graph.set_control_events(options.control_events);
        graph.set_keep_history(options.keep_history);
        Ok(Self {
            graph,
            file: None,
            writable: true,
            path: Some(path.to_path_buf()),
        })
    }

    /// Open a directory-backed database. A missing journal is [`ErrorKind::NotFound`].
    pub fn open_directory(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let journal = path.join(REDCORE_JOURNAL_FILE);
        if !journal.is_file() {
            return Err(Error::new(
                ErrorKind::NotFound,
                "not_found",
                format!("{} does not exist", journal.display()),
            ));
        }
        let mut graph =
            RedCoreGraphStore::open(path, directory_options()).map_err(Error::from_storage)?;
        graph
            .publish_rebuilt_fulltext()
            .map_err(Error::from_storage)?;
        Ok(Self {
            graph,
            file: None,
            writable: true,
            path: Some(path.to_path_buf()),
        })
    }

    /// Copy a directory journal into a new `.RR` file without modifying the source.
    pub fn migrate_journal(
        source_journal: impl AsRef<Path>,
        target: impl AsRef<Path>,
        options: &OpenOptions,
    ) -> Result<MigrationReport, Error> {
        let device = DeviceId::new(&options.device_name);
        let receipt =
            migrate_journal_to_rr(source_journal, target, &device).map_err(Error::from_storage)?;
        Ok(MigrationReport {
            source_preserved: receipt.source_preserved,
            chunks_copied: receipt.chunks_copied,
            named_roots: receipt.named_roots,
            database_id: receipt.database.as_str().to_string(),
        })
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn is_writable(&self) -> bool {
        self.writable
    }

    /// The engine this handle is using. Write methods on it are not available
    /// through a shared reference. [`Self::engine_mut`] refuses a read-only open.
    pub fn engine(&self) -> &RedCoreGraphStore {
        &self.graph
    }

    pub fn engine_mut(&mut self) -> Result<&mut RedCoreGraphStore, Error> {
        self.require_writable()?;
        Ok(&mut self.graph)
    }

    pub fn stats(&self) -> Result<GraphStats, Error> {
        self.graph.stats().map_err(Error::from_storage)
    }

    pub fn inspect(&self) -> Result<Inspection, Error> {
        let stats = self.stats()?;
        Ok(Inspection {
            path: self.path.as_ref().map(|path| path.display().to_string()),
            database_id: self
                .file
                .as_ref()
                .map(|file| file.database().as_str().to_string()),
            writable: self.writable,
            version: stats.version,
            nodes: stats.nodes_total,
            edges: stats.edges_total,
        })
    }

    /// Check the recovered graph. Does not rebuild indexes and does not write.
    pub fn verify(&self) -> Result<VerifyReport, Error> {
        self.graph.verify().map_err(Error::from_storage)
    }

    pub fn vector_designation_count(&self) -> usize {
        self.graph.vector_designations().len()
    }

    pub fn designate_vector(
        &mut self,
        label: &str,
        property: &str,
        dimension: usize,
    ) -> Result<(), Error> {
        self.require_writable()?;
        self.graph
            .designate_vector_property(label, property, dimension)
            .map_err(Error::from_storage)
    }

    /// Keep a full-text index of `fields` of every node labelled `label`, each
    /// field's score scaled by its boost. The index is saved in the database
    /// with each commit and read back on open; it is rebuilt from the nodes
    /// only when a commit it missed is found, and a writable open then saves
    /// the rebuilt index. Designating the same fields again changes nothing.
    #[cfg(feature = "fulltext")]
    pub fn designate_fulltext(&mut self, label: &str, fields: &[(&str, f32)]) -> Result<(), Error> {
        self.require_writable()?;
        self.graph
            .designate_fulltext(label, fields)
            .map_err(Error::from_storage)
    }

    /// Rank `label` nodes by BM25 over the designated fields. Every word of
    /// `query` must match one field, the last word also as the start of a
    /// longer word, and a word that splits into several terms matches as a
    /// phrase. `within` keeps only the given node ids. Each hit carries an
    /// HTML-escaped snippet with its matches in `<mark>`.
    #[cfg(feature = "fulltext")]
    pub fn fulltext_search(
        &self,
        label: &str,
        query: &str,
        limit: usize,
        within: Option<&[String]>,
    ) -> Result<Vec<FullTextHit>, Error> {
        self.graph
            .fulltext_search(label, query, limit, within)
            .map_err(Error::from_storage)
    }

    pub fn mutate(&mut self, batch: GraphMutationBatch) -> Result<GraphTransaction, Error> {
        self.require_writable()?;
        self.graph.commit_batch(batch).map_err(Error::from_storage)
    }

    pub fn upsert_node(&mut self, node: NodeRecord) -> Result<(), Error> {
        self.require_writable()?;
        self.graph.upsert_node(node).map_err(Error::from_storage)?;
        Ok(())
    }

    /// Store `body` as a content-addressed file payload and commit a `File`
    /// node whose manifest points at it. The bytes are the caller's.
    pub fn put_file_node(
        &mut self,
        id: &str,
        logical_path: &str,
        body: &[u8],
    ) -> Result<(), Error> {
        self.require_writable()?;
        let publication_key = format!("graph-node:{id}:property:{FILE_CONTENT_MANIFEST_PROPERTY}");
        let store = self.graph.unified_chunk_store();
        let staged = stage_file_payload(store.as_ref(), &publication_key, body)
            .map_err(Error::from_storage)?;
        let reference = staged.reference().clone();
        let mut properties = serde_json::json!({ "path": logical_path });
        install_file_manifest_reference(
            properties.as_object_mut().expect("path object"),
            &reference,
            0,
        )
        .map_err(Error::from_storage)?;
        self.graph
            .upsert_node(NodeRecord::new(id, ["File"], properties))
            .map_err(Error::from_storage)?;
        staged.finish_after_publication();
        Ok(())
    }

    /// Read a file payload back from the manifest on `node_id`.
    pub fn read_file_node(&self, node_id: &str) -> Result<Vec<u8>, Error> {
        let node = self
            .graph
            .get_node(node_id)
            .map_err(Error::from_storage)?
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::NotFound,
                    "not_found",
                    format!("no node {node_id}"),
                )
            })?;
        let value = node
            .properties
            .get(FILE_CONTENT_MANIFEST_PROPERTY)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Storage,
                    "file_manifest_missing",
                    format!("{node_id} has no file manifest"),
                )
            })?;
        let reference = FileManifestRef::from_value(value).map_err(Error::from_storage)?;
        let store = self.graph.unified_chunk_store();
        read_file_payload(store.as_chunk_store(), &reference).map_err(Error::from_storage)
    }

    /// Rewrite the `.RR` file without the chunks no committed root reaches,
    /// such as superseded catalog roots, journal frames and payloads nothing
    /// names any more. Every recorded revision stays readable.
    ///
    /// The walk and the rewrite happen under the file's writer claim, and the
    /// rewrite is refused, leaving the file as it was, if a commit lands in
    /// between. A directory database or read-only handle is refused.
    pub fn compact(&mut self) -> Result<CompactionReport, Error> {
        self.require_writable()?;
        let file = self.file.as_ref().ok_or_else(|| {
            Error::new(
                ErrorKind::Storage,
                "compaction_unsupported",
                "only a native .RR file can be compacted",
            )
        })?;
        compact_committed_reachable(file.store().as_ref(), &CoreReachabilityCodec)
            .map(|report| report.compaction)
            .map_err(Error::from_storage)
    }

    /// Diagnostics for crash-safety tests; absent from normal library builds.
    #[cfg(feature = "test-fault-injection")]
    #[doc(hidden)]
    pub fn journal_for_test(&self) -> &rustyred_thg_core::JournalChunkStore {
        self.file
            .as_ref()
            .expect("fault injection requires a native .RR database")
            .store()
            .as_ref()
    }

    fn require_writable(&self) -> Result<(), Error> {
        if self.writable {
            Ok(())
        } else {
            Err(Error::new(
                ErrorKind::ReadOnly,
                "read_only",
                "this database was opened read-only",
            ))
        }
    }
}

fn require_existing(path: &Path) -> Result<(), Error> {
    if path.is_file() {
        Ok(())
    } else {
        Err(Error::new(
            ErrorKind::NotFound,
            "not_found",
            format!("{} does not exist", path.display()),
        ))
    }
}

fn file_options() -> RedCoreOptions {
    RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 100,
        strict_acid: false,
    }
}

fn directory_options() -> RedCoreOptions {
    file_options()
}

fn open_graph(
    store: Arc<impl UnifiedChunkStore + 'static>,
    options: &OpenOptions,
) -> Result<RedCoreGraphStore, Error> {
    let store: Arc<dyn UnifiedChunkStore> = store;
    let mut graph = RedCoreGraphStore::open_with_chunk_store(store, file_options())
        .map_err(Error::from_storage)?;
    graph.set_control_events(options.control_events);
    graph.set_keep_history(options.keep_history);
    Ok(graph)
}
