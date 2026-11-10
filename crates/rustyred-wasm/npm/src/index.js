// The asynchronous client for a database held by a dedicated worker.

/** A failed call. `code` is stable for programs, such as `cypher_write_unsupported`. */
export class RustyRedError extends Error {
  constructor({ code, message }) {
    super(message)
    this.name = 'RustyRedError'
    this.code = code
  }
}

const defaultWorker = () => new Worker(new URL('./worker.js', import.meta.url), { type: 'module' })

export class Database {
  #worker
  #pending = new Map()
  #listeners = new Map()
  #next = 1
  #closed = false

  /** @internal Use `open` or `inMemory`. */
  constructor(worker) {
    this.#worker = worker
    worker.addEventListener('message', ({ data }) => {
      if (data.subscription !== undefined) {
        const listener = this.#listeners.get(data.subscription)
        if (!listener) return
        if (data.error) listener.error?.(new RustyRedError(data.error))
        else listener.next(data.value)
        return
      }
      const settle = this.#pending.get(data.id)
      if (!settle) return
      this.#pending.delete(data.id)
      if (data.ok) settle.resolve(data.value)
      else settle.reject(new RustyRedError(data.error))
    })
    worker.addEventListener('error', (event) => {
      const error = new RustyRedError({
        code: 'worker_failure',
        message: event.message || 'the database worker stopped',
      })
      for (const settle of this.#pending.values()) settle.reject(error)
      this.#pending.clear()
    })
  }

  /** @internal */
  _send(message) {
    if (this.#closed) {
      return Promise.reject(
        new RustyRedError({ code: 'database_closed', message: 'the database is closed' }),
      )
    }
    const id = this.#next++
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject })
      this.#worker.postMessage({ ...message, id })
    })
  }

  #call(method, args) {
    return this._send({ op: 'call', method, args })
  }

  inspect() {
    return this.#call('inspect')
  }
  verify() {
    return this.#call('verify')
  }
  mutate(batch) {
    return this.#call('mutate', batch)
  }
  getNode(id) {
    return this.#call('getNode', { id })
  }
  getEdge(id) {
    return this.#call('getEdge', { id })
  }
  queryNodes(filter = {}) {
    return this.#call('queryNodes', filter)
  }
  neighbors(filter) {
    return this.#call('neighbors', filter)
  }
  query(cypher, params, rowCap) {
    return this.#call('query', { cypher, params: params ?? {}, rowCap })
  }

  /**
   * Keep a read current. `next` receives its result now and again after any
   * commit that changes it. Returns the function that stops it.
   */
  subscribe(read, next, error) {
    const subscription = this.#next++
    this.#listeners.set(subscription, { next, error })
    this._send({ op: 'subscribe', subscription, method: read.method, args: read.args }).catch(
      (failure) => {
        this.#listeners.delete(subscription)
        error?.(failure)
      },
    )
    return () => {
      if (!this.#listeners.delete(subscription) || this.#closed) return
      this._send({ op: 'unsubscribe', subscription }).catch(() => {})
    }
  }

  /** Release the database and stop its worker. Later calls reject with `database_closed`. */
  async close() {
    if (this.#closed) return
    try {
      await this.#call('close')
    } finally {
      this.#closed = true
      this.#listeners.clear()
      this.#worker.terminate()
    }
  }
}

async function start(message, settings = {}) {
  const worker = settings.worker ?? defaultWorker()
  const database = new Database(worker)
  try {
    await database._send({ ...message, wasm: settings.wasm })
  } catch (error) {
    worker.terminate()
    throw error
  }
  return database
}

/**
 * Open, creating when absent, the database named `name` in this origin's
 * private file system. One page holds a database at a time.
 */
export function open(name, settings = {}) {
  const { worker, wasm, ...options } = settings
  return start({ op: 'open', name, options }, { worker, wasm })
}

/** An empty graph with no file and no durability. */
export function inMemory(settings = {}) {
  return start({ op: 'inMemory' }, settings)
}

/** Delete the database named `name`. It must not be open. */
export async function remove(name) {
  const root = await navigator.storage.getDirectory()
  try {
    await root.removeEntry(name, { recursive: true })
  } catch (error) {
    if (error?.name !== 'NotFoundError') throw error
  }
}
