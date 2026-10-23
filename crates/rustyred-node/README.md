# @rustyred/node

Embedded Rusty Red for Node.js and Bun. A `Database` is one native `.rr` file (or a graph in memory) opened in the calling process. There is no server.

```js
const { Database } = require('@rustyred/node')

const db = Database.create('notes.rr', { controlEvents: false, keepHistory: false })
db.designateFulltext('Note', { title: 5, body: 1 })
db.mutate({
  upsertNodes: [{ id: 'note:a', labels: ['Note'], properties: { title: 'Garden', body: 'green' } }],
})
db.fulltextSearch('Note', 'gard', 10) // [{ id: 'note:a', score, snippet: '<mark>Garden</mark>' }]
db.close()
```

Calls are synchronous and run on the calling thread. A failure throws an `Error` whose `code` is the database's stable error code, such as `rr_already_exists`, `not_found`, `read_only` or `fulltext_not_designated`.

- `mutate(batch)` commits one batch atomically. Deletes apply before upserts, and deleting a node deletes its edges.
- `getNode`, `getEdge`, `queryNodes` (label, property equality, a string prefix in byte order) and `neighbors` read the graph directly.
- `query(cypher, params)` runs the read-only Cypher dialect described in `docs/api-contract.md`.
- `designateFulltext` and `fulltextSearch` keep a BM25 index in the file.
- `compact()` rewrites the file without chunks nothing reaches.

For derived data, such as an index rebuilt from files, open with `controlEvents: false` and `keepHistory: false`, and call `compact()` now and then. The file then stays near the size of what it holds.

## Building

```text
npm ci
npm run build      # release addon for this platform
npm test
```

The build needs Rust 1.96.1. `npm pack` makes a tarball holding the addon built for this platform.
