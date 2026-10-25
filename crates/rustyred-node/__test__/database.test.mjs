import assert from 'node:assert/strict'
import { mkdtempSync, rmSync, statSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { after, test } from 'node:test'

const { Database } = createRequire(import.meta.url)('../index.js')

const scratch = mkdtempSync(path.join(tmpdir(), 'rustyred-node-'))
after(() => rmSync(scratch, { recursive: true, force: true }))
let files = 0
const fresh = () => path.join(scratch, `db-${files++}.rr`)
const derived = { controlEvents: false, keepHistory: false }

const note = (id, title, body = '') => ({
  id,
  labels: ['Note'],
  properties: { title, body },
})
const ids = (nodes) => nodes.map((node) => node.id)

test('nodes and edges survive reopening, and create and open check the path', () => {
  const file = fresh()
  const db = Database.create(file, derived)
  assert.equal(db.path, file)
  assert.equal(db.writable, true)
  db.mutate({
    upsertNodes: [note('a', 'Alpha'), note('b', 'Beta')],
    upsertEdges: [
      { id: 'a-b', from: 'a', to: 'b', type: 'LINKS', properties: { line: 3 } },
    ],
  })
  db.close()
  assert.throws(() => db.getNode('a'), { code: 'database_closed' })
  assert.throws(() => Database.create(file), { code: 'rr_already_exists' })
  assert.throws(() => Database.open(fresh()), { code: 'not_found' })

  const reopened = Database.open(file, derived)
  assert.deepEqual(reopened.getNode('a'), note('a', 'Alpha'))
  assert.deepEqual(reopened.getEdge('a-b'), {
    id: 'a-b',
    from: 'a',
    to: 'b',
    type: 'LINKS',
    properties: { line: 3 },
  })
  assert.equal(reopened.getNode('missing'), null)
  assert.equal(reopened.inspect().nodes, 2)
  reopened.close()
})

test('a batch deletes before it upserts, and a deleted node takes its edges', () => {
  const db = Database.inMemory()
  db.mutate({
    upsertNodes: [note('a', 'A'), note('b', 'B'), note('c', 'C')],
    upsertEdges: [
      { id: 'a-b', from: 'a', to: 'b', type: 'LINKS', properties: {} },
      { id: 'c-b', from: 'c', to: 'b', type: 'LINKS', properties: {} },
    ],
  })
  // The upsert of `a` is listed first and still lands after its delete.
  const commit = db.mutate({
    upsertNodes: [note('a', 'A again')],
    deleteNodes: ['a', 'b'],
  })
  assert.ok(commit.txnId > 0)
  assert.equal(db.getNode('a').properties.title, 'A again')
  assert.equal(db.getNode('b'), null)
  assert.equal(db.getEdge('a-b'), null)
  assert.equal(db.getEdge('c-b'), null)
  assert.throws(
    () =>
      db.mutate({
        upsertEdges: [{ id: 'x', from: 'a', to: 'gone', type: 'LINKS', properties: {} }],
      }),
    (error) => typeof error.code === 'string' && error.code.length > 0,
  )
})

test('queryNodes filters by label, properties and a string prefix', () => {
  const db = Database.inMemory()
  const entry = (p, kind) => ({
    id: `entry:${p}`,
    labels: ['Entry'],
    properties: { path: p, kind },
  })
  db.mutate({
    upsertNodes: [
      entry('b/z.md', 'file'),
      entry('b', 'directory'),
      entry('a.md', 'file'),
      entry('b2/y.md', 'file'),
      entry('b/a.md', 'file'),
      { id: 'other', labels: ['Other'], properties: { path: 'b/other' } },
    ],
  })
  const byPrefix = (value, extra = {}) =>
    ids(db.queryNodes({ label: 'Entry', prefix: { property: 'path', value }, ...extra }))
  assert.deepEqual(byPrefix(''), [
    'entry:a.md',
    'entry:b',
    'entry:b/a.md',
    'entry:b/z.md',
    'entry:b2/y.md',
  ])
  assert.deepEqual(byPrefix('b/'), ['entry:b/a.md', 'entry:b/z.md'])
  assert.deepEqual(byPrefix('b/', { limit: 1 }), ['entry:b/a.md'])
  assert.deepEqual(byPrefix('', { properties: { kind: 'directory' } }), ['entry:b'])
  assert.deepEqual(
    ids(db.queryNodes({ prefix: { property: 'path', value: 'b/' } })),
    ['entry:b/a.md', 'other', 'entry:b/z.md'],
  )
  assert.deepEqual(
    ids(db.queryNodes({ label: 'Entry', properties: { kind: 'file', path: 'a.md' } })),
    ['entry:a.md'],
  )
})

test('neighbors follow edges in either direction, by type', () => {
  const db = Database.inMemory()
  db.mutate({
    upsertNodes: [note('a', 'A'), note('b', 'B'), { id: 'k', labels: ['Key'], properties: {} }],
    upsertEdges: [
      { id: 'a-k', from: 'a', to: 'k', type: 'LINKS_TO', properties: { line: 2 } },
      { id: 'b-k', from: 'b', to: 'k', type: 'TAGGED', properties: {} },
    ],
  })
  const sources = (edgeType) =>
    db
      .neighbors({ nodeId: 'k', direction: 'in', edgeType })
      .map((hit) => hit.nodeId)
      .sort()
  assert.deepEqual(sources(), ['a', 'b'])
  assert.deepEqual(sources('LINKS_TO'), ['a'])
  assert.deepEqual(db.neighbors({ nodeId: 'a', direction: 'out' }), [
    { edgeId: 'a-k', nodeId: 'k', type: 'LINKS_TO', properties: { line: 2 } },
  ])
})

test('Cypher reads take bound parameters and refuse writes', () => {
  const db = Database.inMemory()
  db.mutate({
    upsertNodes: [note('a', 'A'), note('b', 'B'), { id: 't', labels: ['Tag'], properties: { key: 'x' } }],
    upsertEdges: [
      { id: 'a-t', from: 'a', to: 't', type: 'TAGGED', properties: { tag: 'X' } },
      { id: 'b-t', from: 'b', to: 't', type: 'TAGGED', properties: { tag: 'x' } },
    ],
  })
  const counted = db.query(
    'MATCH (n:Note)-[:TAGGED]->(t:Tag) RETURN t.key AS key, count(n) AS count',
  )
  assert.deepEqual(counted.columns, ['key', 'count'])
  assert.deepEqual(counted.rows, [['x', 2]])
  const titled = db.query('MATCH (n:Note) WHERE n.title = $title RETURN n.title', { title: 'B' })
  assert.deepEqual(titled.rows, [['B']])
  assert.throws(() => db.query("CREATE (n:Note {title: 'x'})"), {
    code: 'cypher_write_unsupported',
  })
})

test('full-text search ranks, filters and marks matches', () => {
  const file = fresh()
  const db = Database.create(file, derived)
  assert.throws(() => db.fulltextSearch('Note', 'x', 5), { code: 'fulltext_not_designated' })
  db.designateFulltext('Note', { title: 5, body: 1 })
  db.mutate({
    upsertNodes: [
      note('body', 'Notes', 'the garden was green'),
      note('title', 'Garden', 'notes'),
      note('other', 'Elsewhere', 'a <b>bold</b> gardener'),
    ],
  })
  const hits = db.fulltextSearch('Note', 'garden', 10)
  // The last word also matches as a prefix, so `gardener` counts, below both.
  assert.deepEqual(ids(hits), ['title', 'body', 'other'])
  assert.ok(hits[0].score > hits[1].score)
  assert.deepEqual(ids(db.fulltextSearch('Note', 'gard', 10, ['other'])), ['other'])
  const [bold] = db.fulltextSearch('Note', 'gardener', 1)
  assert.match(bold.snippet, /&lt;b&gt;bold&lt;\/b&gt; <mark>gardener<\/mark>/)
  db.close()

  const reader = Database.openReadOnly(file)
  assert.equal(reader.writable, false)
  assert.deepEqual(ids(reader.fulltextSearch('Note', 'green', 10)), ['body'])
  assert.throws(() => reader.mutate({ deleteNodes: ['body'] }), { code: 'read_only' })
  reader.close()
})

test('a derived database stays near the size of its data', () => {
  const file = fresh()
  const db = Database.create(file, derived)
  db.designateFulltext('Note', { title: 1, body: 1 })
  for (let round = 0; round < 200; round++)
    db.mutate({ upsertNodes: [note('churn', `round ${round}`, 'x'.repeat(1000))] })
  assert.equal(db.queryNodes({ label: 'ControlEvent' }).length, 0)
  const before = statSync(file).size
  const report = db.compact()
  assert.ok(report.chunksRemoved > 0)
  assert.ok(report.bytesAfter < report.bytesBefore / 4)
  const after = statSync(file).size
  assert.ok(after < before / 4, `${after} of ${before}`)
  assert.deepEqual(ids(db.fulltextSearch('Note', 'round 199', 5)), ['churn'])
  db.close()
})
