// The dedicated worker that owns one database.
//
// OPFS synchronous access handles exist only here, and an open handle is an
// exclusive lock on its file, so one worker holds one database. Every write
// goes through this worker, which is why a subscription can be kept current
// by running its read again after each commit that succeeds.
import init, { Database } from '../pkg/rustyred_wasm.js'

const READS = new Set(['inspect', 'getNode', 'getEdge', 'queryNodes', 'neighbors', 'query'])
const WRITES = new Set(['mutate'])

let database = null
const subscriptions = new Map()

const failure = (error) => ({
  code: typeof error?.code === 'string' ? error.code : 'worker_failure',
  message: error?.message ?? String(error),
})

function publish(id) {
  const subscription = subscriptions.get(id)
  let text
  try {
    text = database.call(subscription.method, subscription.args)
  } catch (error) {
    subscription.last = undefined
    postMessage({ subscription: id, error: failure(error) })
    return
  }
  if (text === subscription.last) return
  subscription.last = text
  postMessage({ subscription: id, value: JSON.parse(text) })
}

async function handle(message) {
  switch (message.op) {
    case 'open':
      await init(message.wasm)
      database = await Database.open(message.name, JSON.stringify(message.options ?? {}))
      return null
    case 'inMemory':
      await init(message.wasm)
      database = Database.inMemory()
      return null
    case 'call': {
      const value = JSON.parse(database.call(message.method, JSON.stringify(message.args ?? null)))
      if (WRITES.has(message.method)) for (const id of subscriptions.keys()) publish(id)
      if (message.method === 'close') {
        database.free()
        database = null
        subscriptions.clear()
      }
      return value
    }
    case 'subscribe':
      if (!READS.has(message.method)) {
        throw Object.assign(new Error(`${message.method} is not a read`), {
          code: 'invalid_subscription',
        })
      }
      subscriptions.set(message.subscription, {
        method: message.method,
        args: JSON.stringify(message.args ?? null),
        last: undefined,
      })
      publish(message.subscription)
      return null
    case 'unsubscribe':
      subscriptions.delete(message.subscription)
      return null
    default:
      throw Object.assign(new Error(`unknown operation ${message.op}`), { code: 'unknown_method' })
  }
}

// Messages are answered in the order they arrive: `open` is asynchronous and
// nothing may reach the database before it settles.
let queue = Promise.resolve()
addEventListener('message', ({ data }) => {
  queue = queue.then(async () => {
    try {
      postMessage({ id: data.id, ok: true, value: await handle(data) })
    } catch (error) {
      postMessage({ id: data.id, ok: false, error: failure(error) })
    }
  })
})
