import { describe, it, expect, beforeAll, afterAll } from 'bun:test'
import { createOpenCoralNode, type OpenCoralNode } from '../../src/p2p/node'
import {
  KV_PROTOCOL,
  registerKVHandler,
  KVSessionClient,
  type KVSessionHandler,
} from '../../src/p2p/kv-protocol'
import { KVChain, SessionPoisonedError } from '../../src/p2p/kv-chain'
import type { VerificationBackend } from '../../src/inference/speculative-session'

describe('KVChain', () => {
  let nodeA: OpenCoralNode
  let nodeB: OpenCoralNode
  let nodeC: OpenCoralNode

  beforeAll(async () => {
    nodeA = await createOpenCoralNode()
    nodeB = await createOpenCoralNode()
    nodeC = await createOpenCoralNode()
    await nodeA.libp2p.dial(nodeB.libp2p.getMultiaddrs()[0])
    await nodeA.libp2p.dial(nodeC.libp2p.getMultiaddrs()[0])
  })

  afterAll(async () => {
    await nodeA.stop()
    await nodeB.stop()
    await nodeC.stop()
  })

  it('forwardAll routes last client to forwardAll, middle client to forward', async () => {
    const nEmbd = 4
    const vocabSize = 8
    const nTokens = 2

    let middleGotForwardAll = false
    let lastGotForwardAll = false

    // nodeB = middle node: receives hidden states, returns transformed hidden states
    const handlerB: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (_sid, input) => {
        return input  // echo hidden states through
      },
      onClose: async () => {},
    }

    // nodeC = last node: receives hidden states, returns logits via forwardAll
    const handlerC: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (_sid, input) => input,
      onClose: async () => {},
      onForwardAll: async (_sid, _input, n) => {
        lastGotForwardAll = true
        return new Float32Array(n * vocabSize).fill(0.5)
      },
    }

    await registerKVHandler(nodeB.libp2p, handlerB)
    await registerKVHandler(nodeC.libp2p, handlerC)

    const clientB = await KVSessionClient.open(nodeA.libp2p, nodeB.libp2p.peerId, 'chain-1', 64)
    const clientC = await KVSessionClient.open(nodeA.libp2p, nodeC.libp2p.peerId, 'chain-1', 64)

    const mockEmbedder = {
      nEmbd,
      embed: async (_ids: Int32Array): Promise<Float32Array> =>
        new Float32Array(nTokens * nEmbd).fill(0.1),
    }

    const chain = new KVChain(mockEmbedder, [clientB, clientC], vocabSize)
    const logits = await chain.forwardAll(new Int32Array([1, 2]))

    expect(logits.length).toBe(nTokens * vocabSize)
    expect(middleGotForwardAll).toBe(false)  // middle used forward, not forwardAll
    expect(lastGotForwardAll).toBe(true)
    for (let i = 0; i < logits.length; i++) expect(logits[i]).toBeCloseTo(0.5, 5)

    await clientB.close()
    await clientC.close()
  })

  it('rollback calls all clients concurrently', async () => {
    let signalB!: () => void
    let signalC!: () => void
    const readyB = new Promise<void>(r => { signalB = r })
    const readyC = new Promise<void>(r => { signalC = r })
    const resolveB: (() => void)[] = []
    const resolveC: (() => void)[] = []
    let calledB = false
    let calledC = false

    const handlerB: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (_sid, input) => input,
      onClose: async () => {},
      onRollback: async () => {
        calledB = true
        signalB()
        await new Promise<void>(r => resolveB.push(r))
      },
    }
    const handlerC: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (_sid, input) => input,
      onClose: async () => {},
      onRollback: async () => {
        calledC = true
        signalC()
        await new Promise<void>(r => resolveC.push(r))
      },
    }

    try { await nodeB.libp2p.unhandle(KV_PROTOCOL) } catch {}
    try { await nodeC.libp2p.unhandle(KV_PROTOCOL) } catch {}
    await registerKVHandler(nodeB.libp2p, handlerB)
    await registerKVHandler(nodeC.libp2p, handlerC)

    const clientB = await KVSessionClient.open(nodeA.libp2p, nodeB.libp2p.peerId, 'rb-chain', 64)
    const clientC = await KVSessionClient.open(nodeA.libp2p, nodeC.libp2p.peerId, 'rb-chain', 64)

    const mockEmbedder = {
      nEmbd: 4,
      embed: async (_ids: Int32Array): Promise<Float32Array> => new Float32Array(4),
    }
    const chain = new KVChain(mockEmbedder, [clientB, clientC], 8)

    // Start rollback — both handlers should be entered before either resolves
    const rollbackPromise = chain.rollback(3)
    await Promise.all([readyB, readyC])  // wait until both handlers have started
    resolveB[0]()
    resolveC[0]()
    await rollbackPromise

    expect(calledB).toBe(true)
    expect(calledC).toBe(true)
    expect(chain.nPast).toBe(3)

    await clientB.close()
    await clientC.close()
  })

  it('nPast increments after forwardAll and decrements after rollback', async () => {
    const nEmbd = 4
    const vocabSize = 8

    const handlerC: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (_sid, input) => input,
      onClose: async () => {},
      onForwardAll: async (_sid, _input, n) => new Float32Array(n * vocabSize),
    }

    try { await nodeC.libp2p.unhandle(KV_PROTOCOL) } catch {}
    await registerKVHandler(nodeC.libp2p, handlerC)

    const clientC = await KVSessionClient.open(nodeA.libp2p, nodeC.libp2p.peerId, 'npast', 64)
    const mockEmbedder = {
      nEmbd,
      embed: async (ids: Int32Array): Promise<Float32Array> => new Float32Array(ids.length * nEmbd),
    }
    const chain = new KVChain(mockEmbedder, [clientC], vocabSize)

    expect(chain.nPast).toBe(0)
    await chain.forwardAll(new Int32Array([1, 2, 3]))
    expect(chain.nPast).toBe(3)
    await chain.rollback(1)
    expect(chain.nPast).toBe(1)

    await clientC.close()
  })

  it('single-client chain uses forwardAll on that client', async () => {
    const vocabSize = 6
    const nEmbd = 4
    let gotForwardAll = false

    const handlerC: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (_sid, input) => input,
      onClose: async () => {},
      onForwardAll: async (_sid, _input, n) => {
        gotForwardAll = true
        return new Float32Array(n * vocabSize).fill(0.1)
      },
    }

    try { await nodeC.libp2p.unhandle(KV_PROTOCOL) } catch {}
    await registerKVHandler(nodeC.libp2p, handlerC)

    const clientC = await KVSessionClient.open(nodeA.libp2p, nodeC.libp2p.peerId, 'single', 64)
    const mockEmbedder = {
      nEmbd,
      embed: async (ids: Int32Array): Promise<Float32Array> => new Float32Array(ids.length * nEmbd),
    }
    const chain = new KVChain(mockEmbedder, [clientC], vocabSize)

    await chain.forwardAll(new Int32Array([1]))
    expect(gotForwardAll).toBe(true)

    await clientC.close()
  })

  it('middle peer failure triggers rollback on preceding peers', async () => {
    const nEmbd = 4
    const vocabSize = 8
    const rollbacksB: number[] = []
    const rollbacksC: number[] = []

    // Peer B succeeds on forward; will be compensated via rollback.
    const handlerB: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (_sid, input) => input,
      onClose: async () => {},
      onRollback: async (_sid, newNPast) => { rollbacksB.push(newNPast) },
    }
    // Peer C is the last node; its forwardAll fails.
    const handlerC: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (_sid, input) => input,
      onForwardAll: async () => { throw new Error('peer C forwardAll failed') },
      onClose: async () => {},
      onRollback: async (_sid, newNPast) => { rollbacksC.push(newNPast) },
    }

    try { await nodeB.libp2p.unhandle(KV_PROTOCOL) } catch {}
    try { await nodeC.libp2p.unhandle(KV_PROTOCOL) } catch {}
    await registerKVHandler(nodeB.libp2p, handlerB)
    await registerKVHandler(nodeC.libp2p, handlerC)

    const clientB = await KVSessionClient.open(nodeA.libp2p, nodeB.libp2p.peerId, 'mid-fail', 64)
    const clientC = await KVSessionClient.open(nodeA.libp2p, nodeC.libp2p.peerId, 'mid-fail', 64)

    const mockEmbedder = {
      nEmbd,
      embed: async (ids: Int32Array) => new Float32Array(ids.length * nEmbd),
    }
    const chain = new KVChain(mockEmbedder, [clientB, clientC], vocabSize)

    let threw = false
    try {
      await chain.forwardAll(new Int32Array([1, 2]))
    } catch {
      threw = true
    }
    expect(threw).toBe(true)
    // Peer B (which advanced) should have been rolled back to preNPast (0)
    expect(rollbacksB).toEqual([0])
    // Peer C's forwardAll failed, so nothing to roll back on C
    expect(rollbacksC).toEqual([])
    expect(chain.nPast).toBe(0)

    await clientB.close().catch(() => {})
    await clientC.close().catch(() => {})
  })

  it('embed failure does not touch peers', async () => {
    const nEmbd = 4
    const vocabSize = 8
    const forwardsB: number[] = []
    const rollbacksB: number[] = []

    const handlerB: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (_sid, input, nTokens) => { forwardsB.push(nTokens); return input },
      onForwardAll: async (_sid, _input, n) => new Float32Array(n * vocabSize),
      onClose: async () => {},
      onRollback: async (_sid, newNPast) => { rollbacksB.push(newNPast) },
    }

    try { await nodeB.libp2p.unhandle(KV_PROTOCOL) } catch {}
    await registerKVHandler(nodeB.libp2p, handlerB)

    const clientB = await KVSessionClient.open(nodeA.libp2p, nodeB.libp2p.peerId, 'embed-fail', 64)

    const failingEmbedder = {
      nEmbd,
      embed: async (_ids: Int32Array): Promise<Float32Array> => {
        throw new Error('embed exploded')
      },
    }
    const chain = new KVChain(failingEmbedder, [clientB], vocabSize)

    let caughtMsg = ''
    try {
      await chain.forwardAll(new Int32Array([1]))
    } catch (err) {
      caughtMsg = (err as Error).message
    }
    expect(caughtMsg).toBe('embed exploded')
    expect(forwardsB).toEqual([])
    expect(rollbacksB).toEqual([])
    expect(chain.nPast).toBe(0)

    await clientB.close()
  })

  it('compensation rollback failure throws SessionPoisonedError', async () => {
    const nEmbd = 4
    const vocabSize = 8

    // B: forward succeeds; rollback fails → compensation fails → poison
    const handlerB: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (_sid, input) => input,
      onClose: async () => {},
      onRollback: async () => { throw new Error('peer B rollback failed') },
    }
    // C is the last node; its forwardAll fails, triggering compensation on B.
    const handlerC: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (_sid, input) => input,
      onForwardAll: async () => { throw new Error('peer C forwardAll failed') },
      onClose: async () => {},
    }

    try { await nodeB.libp2p.unhandle(KV_PROTOCOL) } catch {}
    try { await nodeC.libp2p.unhandle(KV_PROTOCOL) } catch {}
    await registerKVHandler(nodeB.libp2p, handlerB)
    await registerKVHandler(nodeC.libp2p, handlerC)

    const clientB = await KVSessionClient.open(nodeA.libp2p, nodeB.libp2p.peerId, 'poison-1', 64)
    const clientC = await KVSessionClient.open(nodeA.libp2p, nodeC.libp2p.peerId, 'poison-1', 64)

    const mockEmbedder = {
      nEmbd,
      embed: async (ids: Int32Array) => new Float32Array(ids.length * nEmbd),
    }
    const chain = new KVChain(mockEmbedder, [clientB, clientC], vocabSize)

    let caught: unknown = null
    try {
      await chain.forwardAll(new Int32Array([1]))
    } catch (err) {
      caught = err
    }
    expect(caught).toBeInstanceOf(SessionPoisonedError)
    expect((caught as SessionPoisonedError).cause).toBeInstanceOf(Error)
    expect(chain.nPast).toBe(0)

    await clientB.close().catch(() => {})
    await clientC.close().catch(() => {})
  })

  it('nPast advances exactly once on successful forwardAll', async () => {
    const nEmbd = 4
    const vocabSize = 8

    const handlerB: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (_sid, input) => input,
      onClose: async () => {},
    }
    const handlerC: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (_sid, input) => input,
      onForwardAll: async (_sid, _input, n) => new Float32Array(n * vocabSize),
      onClose: async () => {},
    }

    try { await nodeB.libp2p.unhandle(KV_PROTOCOL) } catch {}
    try { await nodeC.libp2p.unhandle(KV_PROTOCOL) } catch {}
    await registerKVHandler(nodeB.libp2p, handlerB)
    await registerKVHandler(nodeC.libp2p, handlerC)

    const clientB = await KVSessionClient.open(nodeA.libp2p, nodeB.libp2p.peerId, 'n-once', 64)
    const clientC = await KVSessionClient.open(nodeA.libp2p, nodeC.libp2p.peerId, 'n-once', 64)

    const mockEmbedder = {
      nEmbd,
      embed: async (ids: Int32Array) => new Float32Array(ids.length * nEmbd),
    }
    const chain = new KVChain(mockEmbedder, [clientB, clientC], vocabSize)

    expect(chain.nPast).toBe(0)
    await chain.forwardAll(new Int32Array([1, 2, 3]))
    expect(chain.nPast).toBe(3)

    await clientB.close()
    await clientC.close()
  })

  it('partial rollback failure throws SessionPoisonedError and does not update nPast', async () => {
    const nEmbd = 4
    const vocabSize = 8

    const handlerB: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (_sid, input) => input,
      onClose: async () => {},
      onRollback: async () => {},  // B rolls back successfully
    }
    const handlerC: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (_sid, input) => input,
      onClose: async () => {},
      onRollback: async () => { throw new Error('peer C rollback failed') },
    }

    try { await nodeB.libp2p.unhandle(KV_PROTOCOL) } catch {}
    try { await nodeC.libp2p.unhandle(KV_PROTOCOL) } catch {}
    await registerKVHandler(nodeB.libp2p, handlerB)
    await registerKVHandler(nodeC.libp2p, handlerC)

    const clientB = await KVSessionClient.open(nodeA.libp2p, nodeB.libp2p.peerId, 'rb-poison', 64)
    const clientC = await KVSessionClient.open(nodeA.libp2p, nodeC.libp2p.peerId, 'rb-poison', 64)

    const mockEmbedder = {
      nEmbd,
      embed: async (ids: Int32Array) => new Float32Array(ids.length * nEmbd),
    }
    const chain = new KVChain(mockEmbedder, [clientB, clientC], vocabSize)

    // Pretend the chain already advanced to position 5 (unit-test the rollback path directly).
    ;(chain as unknown as { nPast: number }).nPast = 5

    let caught: unknown = null
    try {
      await chain.rollback(3)
    } catch (err) {
      caught = err
    }

    expect(caught).toBeInstanceOf(SessionPoisonedError)
    // nPast must stay at 5 because the rollback was partial.
    expect(chain.nPast).toBe(5)

    await clientB.close().catch(() => {})
    await clientC.close().catch(() => {})
  })

  it('forwardOne increments nPast by 1 and routes to last client forwardAll', async () => {
    const vocabSize = 6
    const nEmbd = 4
    let gotForwardAll = false

    const handlerC: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (_sid, input) => input,
      onClose: async () => {},
      onForwardAll: async (_sid, _input, n) => {
        gotForwardAll = true
        return new Float32Array(n * vocabSize).fill(0.7)
      },
    }

    try { await nodeC.libp2p.unhandle(KV_PROTOCOL) } catch {}
    await registerKVHandler(nodeC.libp2p, handlerC)

    const clientC = await KVSessionClient.open(nodeA.libp2p, nodeC.libp2p.peerId, 'fwd-one', 64)
    const mockEmbedder = {
      nEmbd,
      embed: async (ids: Int32Array): Promise<Float32Array> => new Float32Array(ids.length * nEmbd),
    }
    const chain = new KVChain(mockEmbedder, [clientC], vocabSize)

    expect(chain.nPast).toBe(0)
    const logits = await chain.forwardOne(42)
    expect(chain.nPast).toBe(1)
    expect(gotForwardAll).toBe(true)
    expect(logits.length).toBe(vocabSize)

    await clientC.close()
  })
})
