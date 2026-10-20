// W03/V03 oracles for the host-owned IndexedDB lens and opaque storage broker.
//
// This file is `include!`d by `graph/tests.rs`, so imports use a module alias or
// fully qualified paths to avoid colliding with the other graph test shards.

use crate::browser_storage as browser_storage_api;

fn browser_storage_scope(
    partition: &str,
    origin: &str,
    database: &str,
) -> theorem_surface_contracts::desktop::StorageScope {
    theorem_surface_contracts::desktop::StorageScope {
        partition: theorem_surface_contracts::desktop::BrowserPartitionId::new(partition),
        origin: theorem_surface_contracts::desktop::Origin::new(origin),
        database: database.to_owned(),
    }
}

fn browser_storage_string_key(value: &str) -> browser_storage_api::EncodedKey {
    // Golden bytes from the pinned Servo codec: type 0x30, each ASCII UTF-16
    // code unit plus one, and the trailing zero terminator stripped.
    let mut encoded = vec![0x30];
    encoded.extend(value.bytes().map(|byte| byte + 1));
    browser_storage_api::EncodedKey::from_servo_bytes(encoded)
}

fn browser_storage_test_database(
    label: &str,
) -> (
    tempfile::TempDir,
    RustyRed,
    browser_storage_api::BrowserStorageBroker,
) {
    let directory = tempfile::Builder::new()
        .prefix(&format!("browser-storage-{label}-"))
        .tempdir()
        .unwrap();
    let database = RustyRed::create(
        directory.path().join("Travis.RR"),
        &DeviceId::new("browser-storage-tests"),
    )
    .unwrap();
    let graph_bytes = b"browser-storage graph root";
    let graph = Hash::of(graph_bytes);
    database
        .store()
        .write(&[(graph, graph_bytes.as_slice())])
        .unwrap();
    rr_commit_graph_root(&database, graph, 0);
    let broker = browser_storage_api::BrowserStorageBroker::new(database.store().clone());
    (directory, database, broker)
}

fn browser_storage_create_people_schema() -> browser_storage_api::BrowserTransaction {
    browser_storage_api::BrowserTransaction {
        mutations: vec![
            browser_storage_api::BrowserMutation::CreateStore {
                schema: browser_storage_api::ObjectStoreSchema {
                    name: "people".to_owned(),
                    key_path: None,
                    auto_increment: true,
                },
            },
            browser_storage_api::BrowserMutation::CreateIndex {
                store: "people".to_owned(),
                schema: browser_storage_api::IndexSchema {
                    name: "email".to_owned(),
                    key_path: browser_storage_api::KeyPath::String("email".to_owned()),
                    unique: true,
                    multi_entry: false,
                },
                backfill: Vec::new(),
            },
            browser_storage_api::BrowserMutation::CreateStore {
                schema: browser_storage_api::ObjectStoreSchema {
                    name: "audit".to_owned(),
                    key_path: None,
                    auto_increment: false,
                },
            },
            // The same index name in another object store is intentionally
            // legal; identity is (object store, index name).
            browser_storage_api::BrowserMutation::CreateIndex {
                store: "audit".to_owned(),
                schema: browser_storage_api::IndexSchema {
                    name: "email".to_owned(),
                    key_path: browser_storage_api::KeyPath::String("email".to_owned()),
                    unique: false,
                    multi_entry: false,
                },
                backfill: Vec::new(),
            },
            browser_storage_api::BrowserMutation::SetVersion { version: 7 },
        ],
    }
}

fn browser_storage_put_person(
    primary: &str,
    email: &str,
    value: &[u8],
    next_generator: i64,
) -> browser_storage_api::BrowserMutation {
    browser_storage_api::BrowserMutation::Put {
        store: "people".to_owned(),
        key: browser_storage_string_key(primary),
        value: value.to_vec(),
        overwrite: false,
        next_key_generator: Some(next_generator),
        index_updates: vec![browser_storage_api::IndexKeys {
            index: "email".to_owned(),
            keys: vec![browser_storage_string_key(email)],
        }],
    }
}

#[test]
fn browser_storage_transaction_publishes_schema_records_indexes_generator_and_version_once() {
    let (_directory, _database, broker) = browser_storage_test_database("atomic");
    let grant = broker
        .issue_grant(
            browser_storage_scope("personal", "https://a.example", "app"),
            theorem_surface_contracts::desktop::PartitionResidency::Persistent,
        )
        .unwrap();

    let mut transaction = browser_storage_create_people_schema();
    transaction.mutations.extend([
        browser_storage_put_person("1", "alice@example.test", b"Alice", 2),
        browser_storage_put_person("2", "bob@example.test", b"Bob", 3),
    ]);
    let receipt = broker.transact(&grant, &transaction).unwrap();

    assert!(receipt.changed);
    assert!(receipt.publication.is_some());
    let snapshot = broker.database(&grant).unwrap();
    assert_eq!(snapshot.version(), 7);
    assert_eq!(snapshot.object_store_names(), ["audit", "people"]);
    assert_eq!(snapshot.key_generator_current_number("people"), Some(3));
    assert_eq!(snapshot.indexes("people").unwrap()[0].name, "email");
    assert_eq!(snapshot.indexes("audit").unwrap()[0].name, "email");

    let records = broker
        .query(
            &grant,
            "people",
            &browser_storage_api::QueryTarget::Index {
                name: "email".to_owned(),
            },
            &browser_storage_api::EncodedKeyRange::default(),
            None,
        )
        .unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(
        records[0].key,
        browser_storage_string_key("alice@example.test")
    );
    assert_eq!(records[0].primary_key, browser_storage_string_key("1"));
    assert_eq!(records[0].value, b"Alice");
    assert_eq!(records[1].primary_key, browser_storage_string_key("2"));
}

#[test]
fn browser_storage_transaction_program_reads_its_staged_writes_before_publication() {
    let (_directory, _database, broker) = browser_storage_test_database("staged-read");
    let grant = broker
        .issue_grant(
            browser_storage_scope("personal", "https://a.example", "app"),
            theorem_surface_contracts::desktop::PartitionResidency::Persistent,
        )
        .unwrap();
    let key = browser_storage_string_key("queued");
    let execution = broker
        .execute_transaction(
            &grant,
            &browser_storage_api::BrowserTransactionProgram {
                operations: vec![
                    browser_storage_api::BrowserOperation::Mutate {
                        mutation: browser_storage_api::BrowserMutation::CreateStore {
                            schema: browser_storage_api::ObjectStoreSchema {
                                name: "queue".to_owned(),
                                key_path: None,
                                auto_increment: false,
                            },
                        },
                    },
                    browser_storage_api::BrowserOperation::Mutate {
                        mutation: browser_storage_api::BrowserMutation::Put {
                            store: "queue".to_owned(),
                            key: key.clone(),
                            value: b"visible in staged state".to_vec(),
                            overwrite: false,
                            next_key_generator: None,
                            index_updates: Vec::new(),
                        },
                    },
                    browser_storage_api::BrowserOperation::Query {
                        store: "queue".to_owned(),
                        target: browser_storage_api::QueryTarget::ObjectStore,
                        range: browser_storage_api::EncodedKeyRange::only(key.clone()),
                        limit: Some(1),
                    },
                ],
            },
        )
        .unwrap();

    assert!(execution.receipt.publication.is_some());
    assert_eq!(execution.results.len(), 3);
    assert_eq!(
        execution.results[2],
        browser_storage_api::BrowserOperationResult::Records {
            records: vec![browser_storage_api::BrowserRecord {
                key: key.clone(),
                primary_key: key,
                value: b"visible in staged state".to_vec(),
            }],
        }
    );
}

#[test]
fn browser_storage_accepts_empty_indexeddb_database_store_and_index_names() {
    let (_directory, _database, broker) = browser_storage_test_database("empty-names");
    let grant = broker
        .issue_grant(
            browser_storage_scope("personal", "https://a.example", ""),
            theorem_surface_contracts::desktop::PartitionResidency::Persistent,
        )
        .unwrap();

    broker
        .transact(
            &grant,
            &browser_storage_api::BrowserTransaction {
                mutations: vec![
                    browser_storage_api::BrowserMutation::CreateStore {
                        schema: browser_storage_api::ObjectStoreSchema {
                            name: String::new(),
                            key_path: None,
                            auto_increment: false,
                        },
                    },
                    browser_storage_api::BrowserMutation::CreateIndex {
                        store: String::new(),
                        schema: browser_storage_api::IndexSchema {
                            name: String::new(),
                            key_path: browser_storage_api::KeyPath::String("value".to_owned()),
                            unique: false,
                            multi_entry: false,
                        },
                        backfill: Vec::new(),
                    },
                ],
            },
        )
        .unwrap();

    let snapshot = broker.database(&grant).unwrap();
    assert_eq!(snapshot.object_store_names(), [""]);
    assert_eq!(snapshot.indexes("").unwrap()[0].name, "");
}

#[test]
fn browser_storage_overwrite_delete_and_clear_keep_index_state_atomic() {
    let (_directory, _database, broker) = browser_storage_test_database("mutations");
    let grant = broker
        .issue_grant(
            browser_storage_scope("personal", "https://a.example", "app"),
            theorem_surface_contracts::desktop::PartitionResidency::Persistent,
        )
        .unwrap();
    let mut seed = browser_storage_create_people_schema();
    seed.mutations.extend([
        browser_storage_put_person("1", "old@example.test", b"old", 2),
        browser_storage_put_person("2", "delete@example.test", b"delete", 3),
    ]);
    broker.transact(&grant, &seed).unwrap();

    broker
        .transact(
            &grant,
            &browser_storage_api::BrowserTransaction {
                mutations: vec![
                    browser_storage_api::BrowserMutation::Put {
                        store: "people".to_owned(),
                        key: browser_storage_string_key("1"),
                        value: b"new".to_vec(),
                        overwrite: true,
                        next_key_generator: Some(4),
                        index_updates: vec![browser_storage_api::IndexKeys {
                            index: "email".to_owned(),
                            keys: vec![browser_storage_string_key("new@example.test")],
                        }],
                    },
                    browser_storage_api::BrowserMutation::Delete {
                        store: "people".to_owned(),
                        range: browser_storage_api::EncodedKeyRange::only(
                            browser_storage_string_key("2"),
                        ),
                    },
                ],
            },
        )
        .unwrap();

    let indexed = broker
        .query(
            &grant,
            "people",
            &browser_storage_api::QueryTarget::Index {
                name: "email".to_owned(),
            },
            &browser_storage_api::EncodedKeyRange::default(),
            None,
        )
        .unwrap();
    assert_eq!(indexed.len(), 1);
    assert_eq!(
        indexed[0].key,
        browser_storage_string_key("new@example.test")
    );
    assert_eq!(indexed[0].primary_key, browser_storage_string_key("1"));
    assert_eq!(indexed[0].value, b"new");

    broker
        .transact(
            &grant,
            &browser_storage_api::BrowserTransaction {
                mutations: vec![browser_storage_api::BrowserMutation::Clear {
                    store: "people".to_owned(),
                }],
            },
        )
        .unwrap();
    assert!(broker
        .query(
            &grant,
            "people",
            &browser_storage_api::QueryTarget::Index {
                name: "email".to_owned(),
            },
            &browser_storage_api::EncodedKeyRange::default(),
            None,
        )
        .unwrap()
        .is_empty());
    assert_eq!(
        broker
            .database(&grant)
            .unwrap()
            .key_generator_current_number("people"),
        Some(4)
    );
}

#[test]
fn browser_storage_unique_constraint_aborts_the_entire_staged_transaction() {
    let (_directory, database, broker) = browser_storage_test_database("unique-abort");
    let grant = broker
        .issue_grant(
            browser_storage_scope("personal", "https://a.example", "app"),
            theorem_surface_contracts::desktop::PartitionResidency::Persistent,
        )
        .unwrap();
    broker
        .transact(&grant, &browser_storage_create_people_schema())
        .unwrap();
    let before_catalog = database.store().read_root().unwrap();
    let before_roots = broker.root_differential(&grant).unwrap();

    let failed = browser_storage_api::BrowserTransaction {
        mutations: vec![
            browser_storage_put_person("1", "same@example.test", b"first", 2),
            browser_storage_put_person("2", "same@example.test", b"second", 3),
            browser_storage_api::BrowserMutation::SetVersion { version: 99 },
        ],
    };
    let error = broker.transact(&grant, &failed).unwrap_err();

    assert_eq!(error.code, "browser_storage_constraint");
    assert_eq!(database.store().read_root().unwrap(), before_catalog);
    assert_eq!(broker.root_differential(&grant).unwrap(), before_roots);
    let snapshot = broker.database(&grant).unwrap();
    assert_eq!(snapshot.version(), 7);
    assert!(broker
        .query(
            &grant,
            "people",
            &browser_storage_api::QueryTarget::ObjectStore,
            &browser_storage_api::EncodedKeyRange::default(),
            None,
        )
        .unwrap()
        .is_empty());
}

#[test]
fn browser_storage_origin_and_partition_cells_are_isolated_and_reuse_sibling_roots() {
    let (_directory, _database, broker) = browser_storage_test_database("isolation");
    let scopes = [
        browser_storage_scope("personal", "https://a.example", "app"),
        browser_storage_scope("personal", "https://b.example", "app"),
        browser_storage_scope("work", "https://a.example", "app"),
        browser_storage_scope("work", "https://b.example", "app"),
    ];
    let grants = scopes
        .iter()
        .map(|scope| {
            broker
                .issue_grant(
                    scope.clone(),
                    theorem_surface_contracts::desktop::PartitionResidency::Persistent,
                )
                .unwrap()
        })
        .collect::<Vec<_>>();

    for (index, grant) in grants.iter().enumerate() {
        broker
            .transact(
                grant,
                &browser_storage_api::BrowserTransaction {
                    mutations: vec![
                        browser_storage_api::BrowserMutation::CreateStore {
                            schema: browser_storage_api::ObjectStoreSchema {
                                name: "same-store".to_owned(),
                                key_path: None,
                                auto_increment: false,
                            },
                        },
                        browser_storage_api::BrowserMutation::Put {
                            store: "same-store".to_owned(),
                            key: browser_storage_string_key("same-key"),
                            value: format!("cell-{index}").into_bytes(),
                            overwrite: false,
                            next_key_generator: None,
                            index_updates: Vec::new(),
                        },
                    ],
                },
            )
            .unwrap();
    }

    for (index, grant) in grants.iter().enumerate() {
        let records = broker
            .query(
                grant,
                "same-store",
                &browser_storage_api::QueryTarget::ObjectStore,
                &browser_storage_api::EncodedKeyRange::default(),
                None,
            )
            .unwrap();
        assert_eq!(records[0].value, format!("cell-{index}").into_bytes());
    }

    let personal_b_before = broker.root_differential(&grants[1]).unwrap();
    let work_a_before = broker.root_differential(&grants[2]).unwrap();
    broker
        .clear_origin(&scopes[0].partition, &scopes[0].origin)
        .unwrap()
        .expect("origin existed");
    let personal_b_after = broker.root_differential(&grants[1]).unwrap();
    let work_a_after = broker.root_differential(&grants[2]).unwrap();

    assert!(broker
        .database(&grants[0])
        .unwrap()
        .object_store_names()
        .is_empty());
    assert_eq!(personal_b_before.origin, personal_b_after.origin);
    assert_eq!(personal_b_before.database, personal_b_after.database);
    assert_ne!(personal_b_before.partition, personal_b_after.partition);
    assert_eq!(work_a_before.partition, work_a_after.partition);
    assert_eq!(work_a_before.origin, work_a_after.origin);
    assert_eq!(work_a_before.database, work_a_after.database);
}

#[test]
fn browser_storage_forged_cross_scope_and_cross_residency_grants_fail_closed() {
    let (_directory, _database, broker) = browser_storage_test_database("capability");
    let scope = browser_storage_scope("personal", "https://a.example", "app");
    let grant = broker
        .issue_grant(
            scope.clone(),
            theorem_surface_contracts::desktop::PartitionResidency::Persistent,
        )
        .unwrap();
    let forged = theorem_surface_contracts::desktop::StorageGrantId::new("forged");

    let forged_error = broker.database(&forged).unwrap_err();
    assert_eq!(forged_error.code, "browser_storage_grant_refused");
    for (wrong_scope, wrong_residency) in [
        (
            browser_storage_scope("personal", "https://b.example", "app"),
            theorem_surface_contracts::desktop::PartitionResidency::Persistent,
        ),
        (
            browser_storage_scope("work", "https://a.example", "app"),
            theorem_surface_contracts::desktop::PartitionResidency::Persistent,
        ),
        (
            scope.clone(),
            theorem_surface_contracts::desktop::PartitionResidency::Ephemeral,
        ),
    ] {
        let error = broker
            .resolve_grant_for(&grant, &wrong_scope, wrong_residency)
            .unwrap_err();
        assert_eq!(error, forged_error);
        assert!(!error.message.contains("a.example"));
        assert!(!error.message.contains("personal"));
        assert!(!error.message.contains("app"));
    }
}

#[test]
fn browser_storage_ephemeral_partition_never_changes_rr_and_destruction_revokes_it() {
    let (_directory, database, broker) = browser_storage_test_database("ephemeral");
    let persistent = broker
        .issue_grant(
            browser_storage_scope("personal", "https://a.example", "app"),
            theorem_surface_contracts::desktop::PartitionResidency::Persistent,
        )
        .unwrap();
    broker
        .transact(
            &persistent,
            &browser_storage_api::BrowserTransaction {
                mutations: vec![browser_storage_api::BrowserMutation::SetVersion { version: 1 }],
            },
        )
        .unwrap();
    let durable_before = database.store().read_root().unwrap();

    let private_partition = theorem_surface_contracts::desktop::BrowserPartitionId::new("private");
    let ephemeral = broker
        .issue_grant(
            browser_storage_scope("private", "https://a.example", "app"),
            theorem_surface_contracts::desktop::PartitionResidency::Ephemeral,
        )
        .unwrap();
    let receipt = broker
        .transact(
            &ephemeral,
            &browser_storage_api::BrowserTransaction {
                mutations: vec![browser_storage_api::BrowserMutation::SetVersion { version: 88 }],
            },
        )
        .unwrap();

    assert_eq!(receipt.publication, None);
    assert_eq!(broker.database(&ephemeral).unwrap().version(), 88);
    assert_eq!(
        broker.root_differential(&ephemeral).unwrap(),
        Default::default()
    );
    assert_eq!(database.store().read_root().unwrap(), durable_before);
    assert_eq!(
        broker
            .destroy_ephemeral_partition(&private_partition)
            .unwrap(),
        1
    );
    assert_eq!(
        broker.database(&ephemeral).unwrap_err().code,
        "browser_storage_grant_refused"
    );
    assert_eq!(broker.database(&persistent).unwrap().version(), 1);
    assert_eq!(database.store().read_root().unwrap(), durable_before);
}

#[derive(Clone)]
struct BrowserStoragePublicationProbe {
    inner: std::sync::Arc<JournalChunkStore>,
    enabled: std::sync::Arc<std::sync::atomic::AtomicBool>,
    active_read_publish_windows: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    maximum_overlap: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl crate::chunk_store::ChunkStore for BrowserStoragePublicationProbe {
    fn get(&self, address: Hash) -> GraphStoreResult<Option<std::sync::Arc<[u8]>>> {
        self.inner.get(address)
    }

    fn has(&self, address: Hash) -> GraphStoreResult<bool> {
        self.inner.has(address)
    }

    fn missing(&self, addresses: &[Hash]) -> GraphStoreResult<Vec<Hash>> {
        self.inner.missing(addresses)
    }

    fn write(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<()> {
        self.inner.write(chunks)
    }

    fn commit(&self, root: Hash, meta: CommitMeta) -> GraphStoreResult<Hash> {
        self.inner.commit(root, meta)
    }

    fn read_root(&self) -> GraphStoreResult<Option<Hash>> {
        if self.enabled.load(std::sync::atomic::Ordering::SeqCst) {
            let active = self
                .active_read_publish_windows
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                + 1;
            self.maximum_overlap
                .fetch_max(active, std::sync::atomic::Ordering::SeqCst);
            // Make overlap deterministic if two brokers can enter the
            // read-build-publish window at the same time.
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        self.inner.read_root()
    }
}

impl crate::chunk_store::ChunkStoreCatalog for BrowserStoragePublicationProbe {
    fn commit_catalog_updates(
        &self,
        updates: &std::collections::BTreeMap<String, Option<Hash>>,
        meta: CommitMeta,
    ) -> GraphStoreResult<crate::chunk_store::CatalogCommitResult> {
        let result = self.inner.commit_catalog_updates(updates, meta);
        if self.enabled.load(std::sync::atomic::Ordering::SeqCst) {
            self.active_read_publish_windows
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        }
        result
    }
}

#[test]
fn browser_storage_multiple_brokers_serialize_the_whole_root_publication_window() {
    let (_directory, database, _) = browser_storage_test_database("concurrent-brokers");
    let enabled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let active_read_publish_windows = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let maximum_overlap = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let store = std::sync::Arc::new(BrowserStoragePublicationProbe {
        inner: database.store().clone(),
        enabled: enabled.clone(),
        active_read_publish_windows,
        maximum_overlap: maximum_overlap.clone(),
    });
    let left_scope = browser_storage_scope("personal", "https://left.example", "app");
    let right_scope = browser_storage_scope("work", "https://right.example", "app");
    let left_broker = browser_storage_api::BrowserStorageBroker::new(store.clone());
    let right_broker = browser_storage_api::BrowserStorageBroker::new(store);
    let left_grant = left_broker
        .issue_grant(
            left_scope.clone(),
            theorem_surface_contracts::desktop::PartitionResidency::Persistent,
        )
        .unwrap();
    let right_grant = right_broker
        .issue_grant(
            right_scope.clone(),
            theorem_surface_contracts::desktop::PartitionResidency::Persistent,
        )
        .unwrap();
    let start = std::sync::Arc::new(std::sync::Barrier::new(2));

    let left_start = start.clone();
    let left = std::thread::spawn(move || {
        left_start.wait();
        left_broker
            .transact(
                &left_grant,
                &browser_storage_api::BrowserTransaction {
                    mutations: vec![browser_storage_api::BrowserMutation::SetVersion {
                        version: 11,
                    }],
                },
            )
            .unwrap();
    });
    let right = std::thread::spawn(move || {
        start.wait();
        right_broker
            .transact(
                &right_grant,
                &browser_storage_api::BrowserTransaction {
                    mutations: vec![browser_storage_api::BrowserMutation::SetVersion {
                        version: 22,
                    }],
                },
            )
            .unwrap();
    });
    left.join().unwrap();
    right.join().unwrap();

    enabled.store(false, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(maximum_overlap.load(std::sync::atomic::Ordering::SeqCst), 1);
    let verifier = browser_storage_api::BrowserStorageBroker::new(database.store().clone());
    for (scope, expected_version) in [(left_scope, 11), (right_scope, 22)] {
        let grant = verifier
            .issue_grant(
                scope,
                theorem_surface_contracts::desktop::PartitionResidency::Persistent,
            )
            .unwrap();
        assert_eq!(
            verifier.database(&grant).unwrap().version(),
            expected_version
        );
    }
}

#[derive(Clone)]
struct BrowserStorageFailPublishStore {
    inner: std::sync::Arc<JournalChunkStore>,
    failure_stage: std::sync::Arc<std::sync::atomic::AtomicU8>,
}

impl crate::chunk_store::ChunkStore for BrowserStorageFailPublishStore {
    fn get(&self, address: Hash) -> GraphStoreResult<Option<std::sync::Arc<[u8]>>> {
        self.inner.get(address)
    }

    fn has(&self, address: Hash) -> GraphStoreResult<bool> {
        self.inner.has(address)
    }

    fn missing(&self, addresses: &[Hash]) -> GraphStoreResult<Vec<Hash>> {
        self.inner.missing(addresses)
    }

    fn write(&self, chunks: &[(Hash, &[u8])]) -> GraphStoreResult<()> {
        if self.failure_stage.load(std::sync::atomic::Ordering::SeqCst) == 1 {
            return Err(crate::graph::error::GraphStoreError::new(
                "browser_storage_injected_write_failure",
                "injected before staging browser chunks",
            ));
        }
        self.inner.write(chunks)
    }

    fn commit(&self, root: Hash, meta: CommitMeta) -> GraphStoreResult<Hash> {
        self.inner.commit(root, meta)
    }

    fn read_root(&self) -> GraphStoreResult<Option<Hash>> {
        self.inner.read_root()
    }
}

impl crate::chunk_store::ChunkStoreCatalog for BrowserStorageFailPublishStore {
    fn commit_catalog_updates(
        &self,
        updates: &std::collections::BTreeMap<String, Option<Hash>>,
        meta: CommitMeta,
    ) -> GraphStoreResult<crate::chunk_store::CatalogCommitResult> {
        if self.failure_stage.load(std::sync::atomic::Ordering::SeqCst) == 2 {
            return Err(crate::graph::error::GraphStoreError::new(
                "browser_storage_injected_publish_failure",
                "injected immediately before RootCatalog publication",
            ));
        }
        self.inner.commit_catalog_updates(updates, meta)
    }
}

#[test]
fn browser_storage_failed_origin_clear_preserves_the_ephemeral_overlay() {
    let (_directory, database, _) = browser_storage_test_database("clear-origin-failure");
    let failure_stage = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0));
    let broker = browser_storage_api::BrowserStorageBroker::new(std::sync::Arc::new(
        BrowserStorageFailPublishStore {
            inner: database.store().clone(),
            failure_stage: failure_stage.clone(),
        },
    ));
    let scope = browser_storage_scope("personal", "https://a.example", "app");
    let persistent = broker
        .issue_grant(
            scope.clone(),
            theorem_surface_contracts::desktop::PartitionResidency::Persistent,
        )
        .unwrap();
    let ephemeral = broker
        .issue_grant(
            scope.clone(),
            theorem_surface_contracts::desktop::PartitionResidency::Ephemeral,
        )
        .unwrap();
    broker
        .transact(&persistent, &browser_storage_create_people_schema())
        .unwrap();
    broker
        .transact(&ephemeral, &browser_storage_create_people_schema())
        .unwrap();
    let catalog_before = database.store().read_root().unwrap();
    let ephemeral_before = broker.database(&ephemeral).unwrap();

    failure_stage.store(2, std::sync::atomic::Ordering::SeqCst);
    let error = broker
        .clear_origin(&scope.partition, &scope.origin)
        .unwrap_err();
    failure_stage.store(0, std::sync::atomic::Ordering::SeqCst);

    assert_eq!(error.code, "browser_storage_injected_publish_failure");
    assert_eq!(database.store().read_root().unwrap(), catalog_before);
    assert_eq!(broker.database(&ephemeral).unwrap(), ephemeral_before);
    assert!(!broker
        .database(&persistent)
        .unwrap()
        .object_store_names()
        .is_empty());
}

#[test]
fn browser_storage_failures_before_root_publication_preserve_the_exact_old_state() {
    let (_directory, database, _) = browser_storage_test_database("publish-failure");
    let failure_stage = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0));
    let store = std::sync::Arc::new(BrowserStorageFailPublishStore {
        inner: database.store().clone(),
        failure_stage: failure_stage.clone(),
    });
    let broker = browser_storage_api::BrowserStorageBroker::new(store);
    let grant = broker
        .issue_grant(
            browser_storage_scope("personal", "https://a.example", "app"),
            theorem_surface_contracts::desktop::PartitionResidency::Persistent,
        )
        .unwrap();
    broker
        .transact(&grant, &browser_storage_create_people_schema())
        .unwrap();
    let before_catalog = database.store().read_root().unwrap();
    let before_roots = broker.root_differential(&grant).unwrap();
    let before_snapshot = broker.database(&grant).unwrap();
    for (stage, expected_code) in [
        (1, "browser_storage_injected_write_failure"),
        (2, "browser_storage_injected_publish_failure"),
    ] {
        failure_stage.store(stage, std::sync::atomic::Ordering::SeqCst);
        let error = broker
            .transact(
                &grant,
                &browser_storage_api::BrowserTransaction {
                    mutations: vec![browser_storage_put_person(
                        "1",
                        "alice@example.test",
                        b"unpublished",
                        2,
                    )],
                },
            )
            .unwrap_err();

        assert_eq!(error.code, expected_code);
        assert_eq!(database.store().read_root().unwrap(), before_catalog);
        assert_eq!(broker.root_differential(&grant).unwrap(), before_roots);
        assert_eq!(broker.database(&grant).unwrap(), before_snapshot);
    }
}

#[test]
fn browser_storage_persistent_state_reopens_from_the_rr_without_a_sidecar() {
    let (directory, database, broker) = browser_storage_test_database("reopen");
    let scope = browser_storage_scope("personal", "https://a.example", "app");
    let grant = broker
        .issue_grant(
            scope.clone(),
            theorem_surface_contracts::desktop::PartitionResidency::Persistent,
        )
        .unwrap();
    broker
        .transact(
            &grant,
            &browser_storage_api::BrowserTransaction {
                mutations: vec![browser_storage_api::BrowserMutation::SetVersion { version: 42 }],
            },
        )
        .unwrap();
    drop(broker);
    drop(database);

    let reopened = RustyRed::open(
        directory.path().join("Travis.RR"),
        &DeviceId::new("browser-storage-reopen"),
    )
    .unwrap();
    let reopened_broker = browser_storage_api::BrowserStorageBroker::new(reopened.store().clone());
    let reopened_grant = reopened_broker
        .issue_grant(
            scope,
            theorem_surface_contracts::desktop::PartitionResidency::Persistent,
        )
        .unwrap();

    assert_eq!(
        reopened_broker.database(&reopened_grant).unwrap().version(),
        42
    );
    let mut files = std::fs::read_dir(directory.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect::<Vec<_>>();
    files.sort();
    assert_eq!(
        files,
        [
            std::ffi::OsString::from("Travis.RR"),
            std::ffi::OsString::from("Travis.RR.lock"),
        ]
    );
}
