// The browser oracle: the database runs in Chromium, in a dedicated worker,
// on OPFS, and what it commits is there after the page is loaded again.
//
// Run `npm run build` first. Needs Playwright with a Chromium it can launch.
import { createServer } from 'node:http'
import { readFile } from 'node:fs/promises'
import { dirname, extname, join, normalize } from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const TYPES = { '.js': 'text/javascript', '.wasm': 'application/wasm', '.html': 'text/html' }
const PAGE = '<!doctype html><meta charset="utf-8"><title>oracle</title>'

const server = createServer(async (request, response) => {
  const path = new URL(request.url, 'http://localhost').pathname
  if (path === '/') return response.writeHead(200, { 'content-type': TYPES['.html'] }).end(PAGE)
  const file = normalize(join(root, path))
  if (!file.startsWith(root)) return response.writeHead(403).end()
  try {
    const body = await readFile(file)
    response.writeHead(200, { 'content-type': TYPES[extname(file)] ?? 'text/plain' }).end(body)
  } catch {
    response.writeHead(404).end()
  }
})
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
const origin = `http://127.0.0.1:${server.address().port}`

const checks = []
const check = (name, pass, detail) => {
  checks.push({ name, pass })
  console.log(`${pass ? 'ok  ' : 'FAIL'} ${name}${detail === undefined ? '' : `: ${JSON.stringify(detail)}`}`)
}

const browser = await chromium.launch()
const context = await browser.newContext()
const page = await context.newPage()
page.on('console', (message) => console.log(`  [page] ${message.text()}`))
page.on('pageerror', (error) => console.log(`  [page error] ${error.message}`))

const load = async () => {
  await page.goto(origin)
  await page.evaluate(async () => {
    window.rr = await import('/src/index.js')
  })
}

try {
  await load()
  await page.evaluate(() => window.rr.remove('oracle'))

  const first = await page.evaluate(async () => {
    const started = performance.now()
    const db = await window.rr.open('oracle')
    const opened = performance.now() - started
    const seen = []
    const stop = db.subscribe(
      { method: 'queryNodes', args: { label: 'Org' } },
      (nodes) => seen.push(nodes.map((node) => node.properties.name).sort()),
    )
    const commit = await db.mutate({
      upsertNodes: [
        { id: 'org:a', labels: ['Org'], properties: { name: 'Alder', area: 'housing' } },
        { id: 'org:b', labels: ['Org'], properties: { name: 'Birch', area: 'transit' } },
      ],
      upsertEdges: [{ id: 'e:1', from: 'org:a', to: 'org:b', type: 'FUNDS', properties: { year: 2026 } }],
    })
    await db.mutate({ upsertNodes: [{ id: 'note:1', labels: ['Note'], properties: { text: 'unrelated' } }] })
    await db.mutate({ upsertNodes: [{ id: 'org:c', labels: ['Org'], properties: { name: 'Cedar', area: 'housing' } }] })
    const rows = await db.query('MATCH (o:Org) WHERE o.area = $area RETURN o.name AS name ORDER BY name', { area: 'housing' })
    let refused
    try {
      await db.query("CREATE (o:Org {name: 'x'})")
    } catch (error) {
      refused = { name: error.name, code: error.code }
    }
    const neighbors = await db.neighbors({ nodeId: 'org:a', direction: 'out' })
    const verify = await db.verify()
    stop()
    await db.mutate({ upsertNodes: [{ id: 'org:d', labels: ['Org'], properties: { name: 'Dogwood' } }] })
    const inspect = await db.inspect()
    await db.close()
    let closed
    try {
      await db.inspect()
    } catch (error) {
      closed = error.code
    }
    return { opened, commit, seen, rows, refused, neighbors, verify, inspect, closed }
  })
  check('opens on OPFS in a worker', first.opened > 0, { ms: Math.round(first.opened) })
  check('a batch commits', first.commit.graphVersion > 0, first.commit)
  check(
    'a subscription sees each change to its read and nothing else',
    JSON.stringify(first.seen) === JSON.stringify([[], ['Alder', 'Birch'], ['Alder', 'Birch', 'Cedar']]),
    first.seen,
  )
  check('cypher reads with bound parameters', JSON.stringify(first.rows.rows) === '[["Alder"],["Cedar"]]', first.rows.rows)
  check('a cypher write is refused by its code', first.refused?.code === 'cypher_write_unsupported', first.refused)
  check('neighbors carry the edge', first.neighbors[0]?.nodeId === 'org:b' && first.neighbors[0]?.properties.year === 2026)
  check('verify passes', first.verify.ok === true)
  check('a closed database refuses calls', first.closed === 'database_closed', first.closed)

  await load()
  const second = await page.evaluate(async () => {
    const started = performance.now()
    const db = await window.rr.open('oracle')
    const opened = performance.now() - started
    const names = (await db.queryNodes({ label: 'Org' })).map((node) => node.properties.name).sort()
    const edge = await db.getEdge('e:1')
    const inspect = await db.inspect()
    const verify = await db.verify()
    let second
    try {
      await window.rr.open('oracle')
    } catch (error) {
      second = { code: error.code, message: error.message }
    }
    await db.close()
    const again = await window.rr.open('oracle')
    const reopened = (await again.inspect()).nodes
    await again.close()
    return { opened, names, edge, inspect, verify, second, reopened }
  })
  check(
    'records are there after a reload',
    JSON.stringify(second.names) === '["Alder","Birch","Cedar","Dogwood"]' && second.edge?.type === 'FUNDS',
    { names: second.names, ms: Math.round(second.opened) },
  )
  check('the version survives a reload', second.inspect.version === first.inspect.version, {
    before: first.inspect.version,
    after: second.inspect.version,
  })
  check('verify passes after a reload', second.verify.ok === true)
  check('a second open of the same database is refused', second.second !== undefined, second.second)
  check('close releases it for the next open', second.reopened === second.inspect.nodes)

  const memory = await page.evaluate(async () => {
    const db = await window.rr.inMemory()
    await db.mutate({ upsertNodes: [{ id: 'n', labels: ['N'], properties: {} }] })
    const found = await db.getNode('n')
    await db.close()
    return found
  })
  check('an in-memory database works', memory?.id === 'n')

  await page.evaluate(() => window.rr.remove('oracle'))

  // A shared database: three pages of one origin, one file.
  const pages = [page, await context.newPage(), await context.newPage()]
  for (const tab of pages) {
    tab.on('pageerror', (error) => console.log(`  [page error] ${error.message}`))
    await tab.goto(origin)
    await tab.evaluate(async () => {
      window.rr = await import('/src/index.js')
      window.seen = []
      window.db = await window.rr.open('shared-oracle', { shared: true })
      window.db.subscribe({ method: 'queryNodes', args: { label: 'Org' } }, (nodes) =>
        window.seen.push(nodes.map((node) => node.properties.name).sort().join()),
      )
    })
  }
  const [tabA, tabB, tabC] = pages
  const add = (tab, id, name) =>
    tab.evaluate(
      ([i, n]) => window.db.mutate({ upsertNodes: [{ id: i, labels: ['Org'], properties: { name: n } }] }),
      [id, name],
    )
  const last = (tab, wanted) =>
    tab
      .waitForFunction((value) => window.seen.at(-1) === value, wanted, { timeout: 10_000 })
      .then(() => true, () => false)
  const owners = async () => {
    const held = []
    for (const tab of pages) if (!tab.isClosed()) held.push(await tab.evaluate(() => window.db.owner))
    return held
  }

  check('three pages open one shared database, and one holds the file',
    (await owners()).filter(Boolean).length === 1, await owners())
  await add(tabB, 'org:a', 'Alder')
  await add(tabC, 'org:b', 'Birch')
  check('a write from any page reaches every page\'s subscription',
    (await last(tabA, "Alder,Birch")) && (await last(tabB, "Alder,Birch")) && (await last(tabC, "Alder,Birch")))
  const sharedRefused = await tabB.evaluate(() =>
    window.db.query("CREATE (o:Org {name: 'x'})").then(() => null, (error) => error.code))
  check("a failure keeps its code across pages", sharedRefused === "cypher_write_unsupported", sharedRefused)

  // The page that holds the file goes away, with its tab.
  const holder = pages[(await owners()).indexOf(true)]
  const others = pages.filter((tab) => tab !== holder)
  await holder.close()
  await add(others[0], 'org:c', 'Cedar')
  check('when the holding page goes, another takes over with the records',
    (await last(others[0], 'Alder,Birch,Cedar')) && (await last(others[1], 'Alder,Birch,Cedar')),
    await owners())
  check('and exactly one of the others now holds the file', (await owners()).filter(Boolean).length === 1)

  // The page that holds it now closes the database, without going away.
  const next = others[(await owners()).indexOf(true)]
  const remaining = others.find((tab) => tab !== next)
  await next.evaluate(() => window.db.close())
  await add(remaining, 'org:d', 'Dogwood')
  check('closing hands it on too', await last(remaining, 'Alder,Birch,Cedar,Dogwood'))
  const afterClose = await next.evaluate(() => window.db.inspect().then(() => null, (error) => error.code))
  check("and the page that closed is refused", afterClose === "database_closed", afterClose)
  await remaining.evaluate(async () => {
    await window.db.close()
    await window.rr.remove('shared-oracle')
  })
} finally {
  await browser.close()
  server.close()
}

const failed = checks.filter((entry) => !entry.pass)
console.log(`\n${checks.length - failed.length} of ${checks.length} checks passed`)
process.exit(failed.length === 0 ? 0 : 1)
