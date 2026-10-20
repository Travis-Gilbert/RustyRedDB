use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

use rustyred_thg_core::{FoldStorage, FoldStorageMutation};

#[test]
fn enabled_floors_match_atomic_ordered_byte_oracle() -> Result<(), Box<dyn Error>> {
    for kind in enabled_floors() {
        let directory = tempfile::tempdir()?;
        let storage = open_floor(kind, directory.path())?;
        run_differential(storage.as_ref())?;
    }
    Ok(())
}

fn run_differential(storage: &dyn FoldStorage) -> Result<(), Box<dyn Error>> {
    let mut oracle = BTreeMap::<(String, Vec<u8>), Vec<u8>>::new();
    let mut random = DeterministicRandom::new(0xF01D_570A_A6E5_11A5);

    for round in 0..128_u16 {
        let mut mutations = Vec::new();
        for ordinal in 0..1 + random.usize(5) {
            let namespace = format!("fold_{}", random.usize(4));
            let key = format!("key:{:02}:{}", random.usize(24), ordinal).into_bytes();
            if random.usize(5) == 0 {
                oracle.remove(&(namespace.clone(), key.clone()));
                mutations.push(FoldStorageMutation::delete(namespace, key));
            } else {
                let value = format!("value:{round}:{ordinal}:{}", random.next()).into_bytes();
                oracle.insert((namespace.clone(), key.clone()), value.clone());
                mutations.push(FoldStorageMutation::put(namespace, key, value));
            }
        }
        let receipt = storage.write_batch(&mutations)?;
        assert_eq!(receipt.floor, storage.floor());
        assert_eq!(receipt.mutations, mutations.len());
        assert!(receipt.immediately_durable);

        for namespace in ["fold_0", "fold_1", "fold_2", "fold_3"] {
            for key_index in 0..24 {
                let prefix = format!("key:{key_index:02}:");
                let expected = oracle
                    .iter()
                    .filter(|((candidate_namespace, key), _)| {
                        candidate_namespace == namespace && key.starts_with(prefix.as_bytes())
                    })
                    .take(3)
                    .map(|((_, key), value)| (key.clone(), value.clone()))
                    .collect::<Vec<_>>();
                let actual = storage
                    .scan_prefix(namespace, prefix.as_bytes(), 3)?
                    .into_iter()
                    .map(|entry| (entry.key, entry.value))
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected, "{} {namespace} {prefix}", storage.floor());
            }
        }
    }

    let atomic_error = storage
        .write_batch(&[
            FoldStorageMutation::put("atomic", b"must_not_land".to_vec(), b"value".to_vec()),
            FoldStorageMutation::put(" invalid ", b"bad".to_vec(), b"value".to_vec()),
        ])
        .unwrap_err();
    assert_eq!(atomic_error.code, "fold_storage_namespace");
    assert_eq!(storage.get("atomic", b"must_not_land")?, None);

    let scan_error = storage.scan_prefix("fold_0", b"", 0).unwrap_err();
    assert_eq!(scan_error.code, "fold_storage_unbounded_scan");
    assert!(storage.disk_bytes()? > 0);
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum FloorKind {
    #[cfg(feature = "redb-floor")]
    Redb,
    #[cfg(feature = "fjall-floor")]
    Fjall,
}

fn enabled_floors() -> Vec<FloorKind> {
    vec![
        #[cfg(feature = "redb-floor")]
        FloorKind::Redb,
        #[cfg(feature = "fjall-floor")]
        FloorKind::Fjall,
    ]
}

fn open_floor(kind: FloorKind, path: &Path) -> Result<Box<dyn FoldStorage>, Box<dyn Error>> {
    match kind {
        #[cfg(feature = "redb-floor")]
        FloorKind::Redb => Ok(Box::new(rustyred_thg_core::RedbFoldStorage::open(path)?)),
        #[cfg(feature = "fjall-floor")]
        FloorKind::Fjall => Ok(Box::new(rustyred_thg_core::FjallFoldStorage::open(path)?)),
    }
}

struct DeterministicRandom(u64);

impl DeterministicRandom {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn usize(&mut self, upper: usize) -> usize {
        (self.next() as usize) % upper
    }
}
