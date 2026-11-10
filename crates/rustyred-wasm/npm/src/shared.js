// One database for every page of an origin.
//
// An OPFS synchronous access handle is exclusive, so only one page can hold a
// database's file. Pages that open a name with `shared: true` agree on which:
// the one that holds the Web Lock for the name is the owner, opens the
// database, and answers the others over a BroadcastChannel. Each page waits in
// the lock's queue, so when the owner closes or its tab goes, the next becomes
// the owner, opens the file, and says so. The others then send again whatever
// had not been answered and whatever they keep subscribed.
//
// A call the old owner committed but did not get to answer is therefore made
// twice. A batch of upserts and deletes commits the same state the second
// time; a caller that needs more than that should not share.
import { RustyRedError } from './index.js'

const CLOSED = { code: 'database_closed', message: 'the database is closed' }

const failureOf = (error) => ({
  code: typeof error?.code === 'string' ? error.code : 'worker_failure',
  message: error?.message ?? String(error),
})

class SharedDatabase {
  #name
  #openExclusive
  #client = crypto.randomUUID()
  #channel
  #leave = new AbortController()
  #release = () => {}
  #local = null
  #closed = false
  #next = 1
  /** Calls not yet answered: id -> { method, args, resolve, reject }. */
  #calls = new Map()
  /** Reads this client keeps current: id -> { read, next, error, stop }. */
  #reads = new Map()
  /** As owner, the reads kept for other clients: "client:id" -> stop. */
  #served = new Map()
  /** As owner, the clients being watched for leaving. */
  #watched = new Set()

  constructor(name, openExclusive) {
    this.#name = name
    this.#openExclusive = openExclusive
    this.#channel = new BroadcastChannel(`rustyred:${name}`)
    this.#channel.addEventListener('message', ({ data }) => this.#receive(data))
    // Held for as long as this client lives, so an owner can tell when it has gone.
    void navigator.locks.request(
      `rustyred:${name}:client:${this.#client}`,
      { signal: this.#leave.signal },
      () => new Promise((resolve) => this.#leave.signal.addEventListener('abort', resolve)),
    ).catch(() => {})
    void navigator.locks
      .request(`rustyred:${name}`, { signal: this.#leave.signal }, () => this.#own())
      .catch(() => {})
  }

  /** This client holds the lock: it opens the file and serves until it closes. */
  async #own() {
    if (this.#closed) return
    try {
      this.#local = await this.#openExclusive()
    } catch (error) {
      // Nothing this page can serve. Let the next in the queue try, and tell
      // this client's own callers why.
      const failure = new RustyRedError(failureOf(error))
      for (const call of this.#calls.values()) call.reject(failure)
      this.#calls.clear()
      for (const read of this.#reads.values()) read.error?.(failure)
      this.#failed = failure
      return
    }
    if (this.#closed) {
      await this.#local.close().catch(() => {})
      return
    }
    for (const [id, call] of this.#calls) {
      this.#calls.delete(id)
      this.#local._call(call.method, call.args).then(call.resolve, call.reject)
    }
    for (const read of this.#reads.values())
      read.stop = this.#local.subscribe(read.read, read.next, read.error)
    this.#channel.postMessage({ kind: 'owner' })
    await new Promise((resolve) => {
      this.#release = resolve
    })
  }

  #failed = null

  #receive(data) {
    if (this.#closed) return
    if (this.#local) return this.#serve(data)
    switch (data.kind) {
      case 'response': {
        if (data.to !== this.#client) return
        const call = this.#calls.get(data.id)
        if (!call) return
        this.#calls.delete(data.id)
        if (data.ok) call.resolve(data.value)
        else call.reject(new RustyRedError(data.error))
        return
      }
      case 'event': {
        if (data.to !== this.#client) return
        const read = this.#reads.get(data.id)
        if (!read) return
        if (data.error) read.error?.(new RustyRedError(data.error))
        else read.next(data.value)
        return
      }
      case 'owner':
        // A new owner knows nothing of what was asked of the last one.
        for (const [id, { method, args }] of this.#calls) this.#post('call', id, { method, args })
        for (const [id, read] of this.#reads) this.#post('subscribe', id, { read: read.read })
    }
  }

  #post(kind, id, fields) {
    this.#channel.postMessage({ kind, from: this.#client, id, ...fields })
  }

  /** As owner: another client's message. */
  #serve(data) {
    const { kind, from, id } = data
    if (!from || from === this.#client) return
    const key = `${from}:${id}`
    switch (kind) {
      case 'call':
        this.#local._call(data.method, data.args).then(
          (value) => this.#channel.postMessage({ kind: 'response', to: from, id, ok: true, value }),
          (error) =>
            this.#channel.postMessage({
              kind: 'response',
              to: from,
              id,
              ok: false,
              error: failureOf(error),
            }),
        )
        return
      case 'subscribe':
        if (this.#served.has(key)) return
        this.#watch(from)
        this.#served.set(
          key,
          this.#local.subscribe(
            data.read,
            (value) => this.#channel.postMessage({ kind: 'event', to: from, id, value }),
            (error) =>
              this.#channel.postMessage({ kind: 'event', to: from, id, error: failureOf(error) }),
          ),
        )
        return
      case 'unsubscribe':
        this.#served.get(key)?.()
        this.#served.delete(key)
    }
  }

  /** As owner: when `client` goes, by closing or with its tab, stop its reads. */
  #watch(client) {
    if (this.#watched.has(client)) return
    this.#watched.add(client)
    void navigator.locks
      .request(
        `rustyred:${this.#name}:client:${client}`,
        { mode: 'shared', signal: this.#leave.signal },
        () => {
        this.#watched.delete(client)
        for (const [key, stop] of this.#served) {
          if (!key.startsWith(`${client}:`)) continue
          stop()
          this.#served.delete(key)
        }
        },
      )
      .catch(() => {})
  }

  _call(method, args) {
    if (this.#closed) return Promise.reject(new RustyRedError(CLOSED))
    if (this.#failed) return Promise.reject(this.#failed)
    if (this.#local) return this.#local._call(method, args)
    const id = this.#next++
    return new Promise((resolve, reject) => {
      const call = { method, args, resolve, reject }
      this.#calls.set(id, call)
      this.#post('call', id, { method, args })
    })
  }

  inspect() {
    return this._call('inspect')
  }
  verify() {
    return this._call('verify')
  }
  mutate(batch) {
    return this._call('mutate', batch)
  }
  getNode(id) {
    return this._call('getNode', { id })
  }
  getEdge(id) {
    return this._call('getEdge', { id })
  }
  queryNodes(filter = {}) {
    return this._call('queryNodes', filter)
  }
  neighbors(filter) {
    return this._call('neighbors', filter)
  }
  query(cypher, params, rowCap) {
    return this._call('query', { cypher, params: params ?? {}, rowCap })
  }

  subscribe(read, next, error) {
    if (this.#closed) {
      error?.(new RustyRedError(CLOSED))
      return () => {}
    }
    const id = this.#next++
    const entry = { read, next, error, stop: undefined }
    this.#reads.set(id, entry)
    if (this.#local) entry.stop = this.#local.subscribe(read, next, error)
    else this.#post('subscribe', id, { read })
    return () => {
      if (!this.#reads.delete(id) || this.#closed) return
      if (entry.stop) entry.stop()
      else this.#post('unsubscribe', id, {})
    }
  }

  /** Whether this page holds the database's file. It can become true at any time. */
  get owner() {
    return this.#local !== null
  }

  /**
   * Leave the database. If this page held its file, it is closed and another
   * page that has it open takes over. Later calls reject with `database_closed`.
   */
  async close() {
    if (this.#closed) return
    this.#closed = true
    const failure = new RustyRedError(CLOSED)
    for (const call of this.#calls.values()) call.reject(failure)
    this.#calls.clear()
    this.#reads.clear()
    for (const stop of this.#served.values()) stop()
    this.#served.clear()
    this.#leave.abort()
    this.#channel.close()
    try {
      await this.#local?.close()
    } finally {
      this.#release()
    }
  }
}

/** @internal `open(name, { shared: true })`. */
export async function openShared(name, openExclusive) {
  const database = new SharedDatabase(name, openExclusive)
  try {
    // Answered by whichever page holds the file, once it has it open.
    await database.inspect()
  } catch (error) {
    await database.close().catch(() => {})
    throw error
  }
  return database
}
