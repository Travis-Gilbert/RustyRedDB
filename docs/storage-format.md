# Storage format

A native database is one file.

- Magic: `THEORRDB` (8 bytes)
- Header length: 4096
- Format version: 1
- Capability version: 1
- Body: the same journal frame stream as `redcore.journal`, starting at offset 4096

`Database::create` writes that file and composes `RedCoreGraphStore` over its chunk store. The graph, its catalog, and content-addressed payloads live in the file. After every writable handle is dropped, copying the file copies the database.

The writer claim is an `flock` on the file plus an in-process owner. A clone of the chunk-store `Arc` keeps the claim. Read-only opens use `ScanMode::ReadOnly` and do not truncate a torn tail. Writable opens repair a torn tail. A second writable open returns `rr_writer_held`.

Reads are pinned to the catalog recovered at open. They are not a live view of another process's later commits.

Directory deployments stay a directory: `redcore.journal` plus whatever payload files that layout already used. `Database::migrate_journal` copies a journal into a new `.RR` file and leaves the source bytes unchanged. It does not migrate unrelated historical AOF distributions.

Compaction refuses a reachable set that omits the current catalog or any named root (`chunk_store_compaction_live_root_omitted`). It does not drop a root it cannot account for.

Temporary lock state is the flock, not a second authoritative file. Do not copy a file while a writer still has it open.

`object_store` is compiled with its `aws` feature because the history publisher names `AmazonS3Builder`. Create, open, query, and verify of a local file do not construct that client.
