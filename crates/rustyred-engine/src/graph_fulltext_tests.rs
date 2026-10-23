use super::*;
use crate::chunk_store::CatalogCommitResult;
use serde_json::json;
use std::sync::atomic::AtomicBool;

fn note(id: &str, title: &str) -> NodeRecord {
    NodeRecord::new(id, ["Note"], json!({ "title": title }))
}

fn hits(store: &RedCoreGraphStore, query: &str) -> Vec<String> {
    store
        .fulltext_search("Note", query, 10, None)
        .unwrap()
        .into_iter()
        .map(|hit| hit.id)
        .collect()
}

fn options() -> RedCoreOptions {
    RedCoreOptions {
        durability: RedCoreDurability::AofAlways,
        snapshot_interval_writes: 0,
        strict_acid: true,
    }
}

#[test]
fn the_committed_index_is_read_back_on_open() -> GraphStoreResult<()> {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut store = RedCoreGraphStore::open(dir.path(), options())?;
        store.designate_fulltext("Note", &[("title", 1.0)])?;
        store.upsert_node(note("a", "Alpha"))?;
        store.upsert_node(note("b", "Beta"))?;
    }
    let mut store = RedCoreGraphStore::open(dir.path(), options())?;
    assert!(!store.fulltext.unpublished());
    let catalog = store.unified_chunk_store().read_root()?;
    store.publish_rebuilt_fulltext()?;
    assert_eq!(store.unified_chunk_store().read_root()?, catalog);
    assert_eq!(hits(&store, "beta"), ["b"]);
    Ok(())
}

#[test]
fn a_commit_the_index_missed_is_rebuilt_on_open() -> GraphStoreResult<()> {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut store = RedCoreGraphStore::open(dir.path(), options())?;
        store.designate_fulltext("Note", &[("title", 1.0)])?;
        store.upsert_node(note("a", "Alpha"))?;
        // What a build without Tantivy does: commit and leave the root alone.
        store.fulltext = FullTextStore::default();
        store.upsert_node(note("b", "Beta"))?;
    }
    let mut store = RedCoreGraphStore::open(dir.path(), options())?;
    assert!(store.fulltext.unpublished());
    assert_eq!(hits(&store, "beta"), ["b"]);
    store.publish_rebuilt_fulltext()?;
    drop(store);

    let store = RedCoreGraphStore::open(dir.path(), options())?;
    assert!(!store.fulltext.unpublished());
    assert_eq!(hits(&store, "beta"), ["b"]);
    Ok(())
}

/// A memory store whose catalog commits can be refused.
struct RefusingCatalog {
    inner: MemChunkStore,
    refuse: AtomicBool,
}

impl ChunkStore for RefusingCatalog {
    fn get(&self, addr: Hash) -> GraphStoreResult<Option<Arc<[u8]>>> {
        self.inner.get(addr)
    }

    fn get_bounded(&self, addr: Hash, max_bytes: usize) -> GraphStoreResult<Option<Arc<[u8]>>> {
        self.inner.get_bounded(addr, max_bytes)
    }

    fn has(&self, addr: Hash) -> GraphStoreResult<bool> {
        self.inner.has(addr)
    }

    fn missing(&self, addrs: &[Hash]) -> GraphStoreResult<Vec<Hash>> {
        self.inner.missing(addrs)
    }

    fn write(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<()> {
        self.inner.write(chunks)
    }

    fn commit(&self, root: Hash, meta: CommitMeta) -> GraphStoreResult<Hash> {
        self.inner.commit(root, meta)
    }

    fn read_root(&self) -> GraphStoreResult<Option<Hash>> {
        self.inner.read_root()
    }
}

impl ChunkStoreCatalog for RefusingCatalog {
    fn commit_catalog_updates(
        &self,
        updates: &BTreeMap<String, Option<Hash>>,
        meta: CommitMeta,
    ) -> GraphStoreResult<CatalogCommitResult> {
        if self.refuse.load(Ordering::SeqCst) {
            return Err(GraphStoreError::new(
                "catalog_refused_for_test",
                "the catalog commit was refused",
            ));
        }
        self.inner.commit_catalog_updates(updates, meta)
    }
}

#[test]
fn a_failed_commit_leaves_the_index_at_the_kept_graph() -> GraphStoreResult<()> {
    let backend = Arc::new(RefusingCatalog {
        inner: MemChunkStore::new(),
        refuse: AtomicBool::new(false),
    });
    let mut store = RedCoreGraphStore::open_with_chunk_store(backend.clone(), options())?;
    store.designate_fulltext("Note", &[("title", 1.0)])?;
    store.upsert_node(note("a", "Alpha"))?;

    backend.refuse.store(true, Ordering::SeqCst);
    assert!(store.upsert_node(note("a", "Gamma")).is_err());
    assert!(store.delete_node("a").is_err());
    backend.refuse.store(false, Ordering::SeqCst);

    assert_eq!(hits(&store, "alpha"), ["a"]);
    assert!(hits(&store, "gamma").is_empty());
    store.upsert_node(note("b", "Beta"))?;
    assert_eq!(hits(&store, "beta"), ["b"]);
    assert_eq!(hits(&store, "alpha"), ["a"]);
    Ok(())
}
