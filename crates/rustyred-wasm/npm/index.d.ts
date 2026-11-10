/** A failed call. `code` is stable for programs, such as `cypher_write_unsupported`. */
export declare class RustyRedError extends Error {
  readonly code: string
}

export interface Node {
  id: string
  labels: Array<string>
  properties: Record<string, unknown>
}

export interface Edge {
  id: string
  from: string
  to: string
  type: string
  properties: Record<string, unknown>
}

/**
 * One atomic commit. Deletes apply before upserts, whatever the order of the
 * lists. Deleting a node also deletes its edges, except those this batch
 * upserts. An upsert replaces the whole record, and an edge's ends must exist
 * once the batch's node upserts are applied.
 */
export interface Batch {
  deleteEdges?: Array<string>
  deleteNodes?: Array<string>
  upsertNodes?: Array<Node>
  upsertEdges?: Array<Edge>
}

export interface Commit {
  txnId: number
  graphVersion: number
}

/** Which nodes `queryNodes` returns. Every given condition must hold. */
export interface NodeFilter {
  label?: string
  /** Properties each node must equal. Strings, numbers, booleans and null are indexed. */
  properties?: Record<string, unknown>
  /** A string property each node's value must start with. The nodes then come in byte order of that value. */
  prefix?: { property: string; value: string }
  limit?: number
}

export interface NeighborFilter {
  nodeId: string
  direction: 'out' | 'in'
  edgeType?: string
}

/** A node next to the one asked about, and the edge between them. */
export interface Neighbor {
  edgeId: string
  nodeId: string
  type: string
  properties: Record<string, unknown>
}

export interface QueryResult {
  columns: Array<string>
  rows: Array<Array<unknown>>
  truncated: boolean
}

export interface Inspection {
  writable: boolean
  version: number
  nodes: number
  edges: number
}

export interface VerifyReport {
  ok: boolean
  stats: Record<string, unknown>
  problems: Array<unknown>
}

/** A read a subscription keeps current. */
export type Read =
  | { method: 'inspect' }
  | { method: 'getNode' | 'getEdge'; args: { id: string } }
  | { method: 'queryNodes'; args?: NodeFilter }
  | { method: 'neighbors'; args: NeighborFilter }
  | { method: 'query'; args: { cypher: string; params?: Record<string, unknown>; rowCap?: number } }

export interface OpenOptions {
  /** A write to a labelled node also commits a `ControlEvent` node recording it. Defaults to true. */
  controlEvents?: boolean
  /** Each commit joins the recorded revision history. Defaults to true. */
  keepHistory?: boolean
}

export interface WorkerSettings {
  /** A worker running `@rustyreddb/browser/worker`, for a bundler that needs to make it itself. */
  worker?: Worker
  /** Where the wasm module is, when it is not beside the package's own glue. */
  wasm?: string | URL
}

export declare class Database {
  private constructor()
  inspect(): Promise<Inspection>
  /** Check the recovered graph without writing. */
  verify(): Promise<VerifyReport>
  mutate(batch: Batch): Promise<Commit>
  getNode(id: string): Promise<Node | null>
  getEdge(id: string): Promise<Edge | null>
  queryNodes(filter?: NodeFilter): Promise<Array<Node>>
  neighbors(filter: NeighborFilter): Promise<Array<Neighbor>>
  /** Run one read-only Cypher query with bound parameters. `rowCap` defaults to 1000. */
  query(cypher: string, params?: Record<string, unknown>, rowCap?: number): Promise<QueryResult>
  /**
   * Keep a read current. `next` receives its result now and again after any
   * commit that changes it. Returns the function that stops it.
   */
  subscribe(read: Read, next: (value: any) => void, error?: (error: RustyRedError) => void): () => void
  /** Release the database and stop its worker. Later calls reject with `database_closed`. */
  close(): Promise<void>
}

/**
 * Open, creating when absent, the database named `name` in this origin's
 * private file system. One page holds a database at a time.
 */
export declare function open(name: string, settings?: OpenOptions & WorkerSettings): Promise<Database>

/** An empty graph with no file and no durability. */
export declare function inMemory(settings?: WorkerSettings): Promise<Database>

/** Delete the database named `name`. It must not be open. */
export declare function remove(name: string): Promise<void>
