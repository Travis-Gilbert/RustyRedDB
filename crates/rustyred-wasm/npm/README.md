# @rustyreddb/browser

Embedded Rusty Red for the browser. A `Database` is one graph in this origin's private file system (OPFS), held by a dedicated worker the package starts. There is no server.

```js
import { open } from '@rustyreddb/browser'

const db = await open('notes')
await db.mutate({
  upsertNodes: [{ id: 'note:a', labels: ['Note'], properties: { title: 'Garden' } }],
})
const stop = db.subscribe({ method: 'queryNodes', args: { label: 'Note' } }, (notes) => {
  console.log(notes.length) // now, and again after a commit that changes the result
})
await db.query('MATCH (n:Note) RETURN n.title AS title') // { columns, rows, truncated }
stop()
await db.close()
```

Every call returns a promise. A failure rejects with a `RustyRedError` whose `code` is the database's stable error code, such as `cypher_write_unsupported` or `database_closed`.

- `mutate(batch)` commits one batch atomically. Deletes apply before upserts, and deleting a node deletes its edges.
- `getNode`, `getEdge`, `queryNodes` (label, property equality, a string prefix in byte order) and `neighbors` read the graph directly.
- `query(cypher, params)` runs the read-only Cypher dialect described in `docs/api-contract.md`.
- `subscribe(read, next)` keeps one of those reads current. The worker runs every subscribed read again after each commit and calls `next` only when the result differs.
- `inMemory()` gives a graph with no file. `remove(name)` deletes a database that is not open.

The names and shapes are those of `@rustyreddb/node`, made asynchronous.

## What it does not do

- **One page holds the file.** The browser gives one exclusive handle to a database's file, so a second plain `open` of the same name, in this tab or another, rejects with `chunk_store_browser_storage` until the first is closed. Open with `shared: true` to use one database from several pages; see below.
- **No full-text search.** `designateFulltext` and `fulltextSearch` are the Node binding's. The engine's index is built with Tantivy, whose default features do not build for `wasm32-unknown-unknown`.
- **No `compact()`.** Compaction rewrites a native `.rr` file.
- **Not a `.rr` file.** The OPFS store is the engine's browser projection. A database made here cannot be opened by the native library, or the reverse.

It needs a browser with OPFS synchronous access handles in workers.

## Several pages, one database

```js
const db = await open('notes', { shared: true })
```

Every page of the origin that opens a name this way uses one database. The page that holds the Web Lock for the name opens the file and answers the others over a `BroadcastChannel`; a write from any page reaches every page's subscriptions. When that page closes the database, or its tab goes, the next page in the lock's queue opens the file and the others send again what had not been answered and what they keep subscribed. `db.owner` says whether this page holds the file now.

A call the last holder committed but did not get to answer is made again by the next. A batch of upserts and deletes commits the same state the second time.

## Bundlers

The client starts its worker with `new Worker(new URL('./worker.js', import.meta.url), { type: 'module' })`, which Vite and webpack 5 follow. To make the worker yourself, pass it, or a function that makes it: `open('notes', { worker: () => new Worker(...) })`, running `@rustyreddb/browser/worker`. A shared database makes its worker only when this page comes to hold the file, so give it the function. To serve the wasm module from somewhere else, pass `wasm`.

## Building

```text
npm run build      # cargo build for wasm32-unknown-unknown, then wasm-bindgen into ./pkg
npm run oracle     # runs the database in headless Chromium: a page reload, and three pages sharing one
```

The build needs Rust 1.96.1 with the `wasm32-unknown-unknown` target and `wasm-bindgen-cli` 0.2.127, the version the workspace pins. The oracle needs Playwright with a Chromium it can launch.
