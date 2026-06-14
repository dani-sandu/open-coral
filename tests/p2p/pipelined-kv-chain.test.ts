import { describe, it, expect } from 'bun:test'
import { PipelinedKVChain } from '../../src/p2p/pipelined-kv-chain'

// Minimal KVChain-shaped stub: only the methods PipelinedKVChain calls.
// We re-use this stub across the file's tests.
function makeFakeChain(opts: {
  forwardAllImpl?: (batch: Int32Array) => Promise<Float32Array>
  rollbackImpl?: (n: number) => Promise<void>
  vocabSize?: number
  clients?: unknown[]
  embedder?: { nEmbd: number; embed: (ids: Int32Array) => Promise<Float32Array> }
} = {}) {
  return {
    vocabSize: opts.vocabSize ?? 8,
    nPast: 0,
    clients: opts.clients ?? [],
    embedder: opts.embedder ?? { nEmbd: 4, embed: async (ids: Int32Array) => new Float32Array(ids.length * 4) },
    forwardAll: opts.forwardAllImpl ?? (async (batch: Int32Array) => new Float32Array(batch.length * 8)),
    forwardOne: async (tokenId: number) => new Float32Array(8),
    rollback: opts.rollbackImpl ?? (async () => {}),
  } as unknown as import('../../src/p2p/kv-chain').KVChain
}

describe('PipelinedKVChain — depth=1 passthrough', () => {
  it('forwardAll delegates byte-identically to underlying chain.forwardAll', async () => {
    const calls: Int32Array[] = []
    const chain = makeFakeChain({
      forwardAllImpl: async (batch) => {
        calls.push(batch)
        return new Float32Array([1, 2, 3, 4, 5, 6, 7, 8])
      },
    })
    const p = new PipelinedKVChain(chain, 1)
    const out = await p.forwardAll(new Int32Array([10, 20, 30]))
    expect(calls.length).toBe(1)
    expect(Array.from(calls[0])).toEqual([10, 20, 30])
    expect(Array.from(out)).toEqual([1, 2, 3, 4, 5, 6, 7, 8])
  })

  it('rollback delegates to chain.rollback at depth=1', async () => {
    const rollbacks: number[] = []
    const chain = makeFakeChain({ rollbackImpl: async (n) => { rollbacks.push(n) } })
    const p = new PipelinedKVChain(chain, 1)
    await p.rollback(42)
    expect(rollbacks).toEqual([42])
  })

  it('vocabSize comes from the underlying chain', () => {
    const chain = makeFakeChain({ vocabSize: 16 })
    const p = new PipelinedKVChain(chain, 1)
    expect(p.vocabSize).toBe(16)
  })
})

describe('PipelinedKVChain — depth=2 happy path', () => {
  it('runs two concurrent submissions across hops with per-hop FIFO', async () => {
    // Synthetic 3-hop chain. Each client.forward records (callOrder, hop) and
    // simulates 10ms of "compute" so we can observe interleaving.
    const callLog: { hop: number; order: number }[] = []
    let nextOrder = 0

    function makeClient(hop: number) {
      return {
        forward: async (hidden: Float32Array, nTokens: number, nEmbd: number) => {
          const order = nextOrder++
          await new Promise(r => setTimeout(r, 10))
          callLog.push({ hop, order })
          // Pass hidden through unchanged (identity).
          return hidden
        },
        forwardAll: async (hidden: Float32Array, nTokens: number, nEmbd: number) => {
          const order = nextOrder++
          await new Promise(r => setTimeout(r, 10))
          callLog.push({ hop, order })
          return new Float32Array(nTokens * 8)
        },
      }
    }

    const clients = [makeClient(0), makeClient(1), makeClient(2)]
    const chain = makeFakeChain({
      clients,
      embedder: { nEmbd: 4, embed: async (ids: Int32Array) => new Float32Array(ids.length * 4) },
      vocabSize: 8,
    })

    const p = new PipelinedKVChain(chain, 2)
    // Two submissions kicked off without intervening await.
    const a = p.forwardAll(new Int32Array([1, 2]))
    const b = p.forwardAll(new Int32Array([3, 4]))
    const [logitsA, logitsB] = await Promise.all([a, b])

    // Each forward must have produced 3 hops = 6 total hop calls.
    expect(callLog.length).toBe(6)
    // Per-hop FIFO: the two calls at each hop happened in submission order.
    const byHop = (h: number) => callLog.filter(c => c.hop === h).map(c => c.order)
    expect(byHop(0).length).toBe(2)
    expect(byHop(1).length).toBe(2)
    expect(byHop(2).length).toBe(2)
    for (const h of [0, 1, 2]) {
      const orders = byHop(h)
      expect(orders[0]).toBeLessThan(orders[1])
    }

    // Optimistic nPast advanced for both immediately.
    expect(p.nPast).toBe(4)
    // Logit lengths.
    expect(logitsA.length).toBe(2 * 8)
    expect(logitsB.length).toBe(2 * 8)
  })

  it('advances optimistic nPast on submit, not resolve', async () => {
    let resolveHop0: ((v: Float32Array) => void) | null = null
    const clients = [
      {
        forward: (h: Float32Array) => new Promise<Float32Array>(r => { resolveHop0 = r }),
        forwardAll: async () => new Float32Array(0),
      },
      {
        forward: async (h: Float32Array) => h,
        forwardAll: async (h: Float32Array, n: number) => new Float32Array(n * 8),
      },
    ]
    const chain = makeFakeChain({
      clients,
      embedder: { nEmbd: 4, embed: async (ids: Int32Array) => new Float32Array(ids.length * 4) },
      vocabSize: 8,
    })
    const p = new PipelinedKVChain(chain, 2)
    const inFlight = p.forwardAll(new Int32Array([1, 2, 3]))
    // nPast advanced even though hop 0 is stuck.
    expect(p.nPast).toBe(3)
    // Flush microtasks so embed resolves and forward() is called (setting resolveHop0).
    await Promise.resolve()
    await Promise.resolve()
    // Unblock and complete.
    resolveHop0!(new Float32Array(12))
    await inFlight
  })
})

describe('PipelinedKVChain — rollback gating', () => {
  it('rollback awaits in-flight forwards before delegating', async () => {
    const events: string[] = []
    let resolveHop0: ((v: Float32Array) => void) | null = null

    const clients = [
      {
        forward: (h: Float32Array) => new Promise<Float32Array>(r => { resolveHop0 = r }),
        forwardAll: async () => new Float32Array(0),
      },
      {
        forward: async (h: Float32Array) => h,
        forwardAll: async (h: Float32Array, n: number) => new Float32Array(n * 8),
      },
    ]
    const chain = makeFakeChain({
      clients,
      embedder: { nEmbd: 4, embed: async (ids: Int32Array) => new Float32Array(ids.length * 4) },
      vocabSize: 8,
      rollbackImpl: async (n) => { events.push(`rollback(${n})`) },
    })
    const p = new PipelinedKVChain(chain, 2)
    const inFlight = p.forwardAll(new Int32Array([1, 2, 3]))
    inFlight.then(() => events.push('forward-resolved'))

    const rollbackPromise = p.rollback(0)
    rollbackPromise.then(() => events.push('rollback-resolved'))

    // Give the event loop a tick — neither should have completed yet.
    await new Promise(r => setTimeout(r, 5))
    expect(events).toEqual([])

    // Unblock the in-flight forward; rollback should now proceed.
    resolveHop0!(new Float32Array(12))
    await rollbackPromise
    await inFlight

    expect(events).toContain('forward-resolved')
    expect(events).toContain('rollback(0)')
    expect(events).toContain('rollback-resolved')
    // Forward MUST resolve before rollback delegates.
    expect(events.indexOf('forward-resolved')).toBeLessThan(events.indexOf('rollback(0)'))
    expect(p.nPast).toBe(0)
  })
})

describe('PipelinedKVChain — error propagation', () => {
  it('propagates error from a later submission while the earlier one resolves cleanly', async () => {
    // Per-hop FIFO means submissions hit hop 0 in submit order. Each call
    // increments callCount; the 2nd call (which is step N+1) throws. Step N
    // (callCount=1) resolves normally, so we verify both: N succeeds, N+1 fails.
    let callCount = 0
    const clients = [
      {
        forward: async (h: Float32Array) => {
          callCount++
          if (callCount === 2) throw new Error('hop0 down for N+1')
          return h
        },
        forwardAll: async () => new Float32Array(0),
      },
      {
        forward: async (h: Float32Array) => h,
        forwardAll: async (h: Float32Array, n: number) => new Float32Array(n * 8),
      },
    ]
    const chain = makeFakeChain({
      clients,
      embedder: { nEmbd: 4, embed: async (ids: Int32Array) => new Float32Array(ids.length * 4) },
      vocabSize: 8,
    })
    const p = new PipelinedKVChain(chain, 2)
    const a = p.forwardAll(new Int32Array([1]))   // step N
    const b = p.forwardAll(new Int32Array([2]))   // step N+1 — errors at hop 0

    await expect(a).resolves.toBeInstanceOf(Float32Array)
    await expect(b).rejects.toThrow('hop0 down for N+1')
  })
})
