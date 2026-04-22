import { describe, it, expect, beforeAll, afterAll } from 'bun:test'
import { createOpenCoralNode, type OpenCoralNode } from '../../src/p2p/node'
import {
  KV_PROTOCOL,
  registerKVHandler,
  KVSessionClient,
  type KVSessionHandler,
} from '../../src/p2p/kv-protocol'

describe('KV session protocol — persistent streams', () => {
  let nodeA: OpenCoralNode
  let nodeB: OpenCoralNode

  beforeAll(async () => {
    nodeA = await createOpenCoralNode()
    nodeB = await createOpenCoralNode()
    await nodeA.libp2p.dial(nodeB.libp2p.getMultiaddrs()[0])
  })

  afterAll(async () => {
    await nodeA.stop()
    await nodeB.stop()
  })

  it('KV_PROTOCOL is correct', () => {
    expect(KV_PROTOCOL).toBe('/opencoral/kv/2.0.0')
  })

  it('open → multiple forwards → close on single stream', async () => {
    const handler: KVSessionHandler = {
      onOpen: async (sessionId) => {
        return { ok: true }
      },
      onForward: async (sessionId, input) => {
        return input  // echo
      },
      onClose: async (sessionId) => {},
    }

    await registerKVHandler(nodeB.libp2p, handler)

    const client = await KVSessionClient.open(
      nodeA.libp2p, nodeB.libp2p.peerId, 'sess-1', 128,
    )

    const input1 = new Float32Array(2 * 4).fill(0.5)
    const out1 = await client.forward(input1, 2, 4)
    expect(out1.length).toBe(8)
    for (let i = 0; i < out1.length; i++) expect(out1[i]).toBeCloseTo(0.5, 5)

    const input2 = new Float32Array(1 * 4).fill(0.9)
    const out2 = await client.forward(input2, 1, 4)
    expect(out2.length).toBe(4)
    for (let i = 0; i < out2.length; i++) expect(out2[i]).toBeCloseTo(0.9, 5)

    await client.close()
  })

  it('forward with requestId passes through', async () => {
    let lastRequestId: string | undefined

    const handler: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (_sessionId, input, _nTokens, _nEmbd, requestId) => {
        lastRequestId = requestId
        return input
      },
      onClose: async () => {},
    }

    try { await nodeB.libp2p.unhandle(KV_PROTOCOL) } catch {}
    await registerKVHandler(nodeB.libp2p, handler)

    const client = await KVSessionClient.open(nodeA.libp2p, nodeB.libp2p.peerId, 'sess-2', 64)
    await client.forward(new Float32Array(4), 1, 4, 'trace-abc')
    await client.close()

    expect(lastRequestId).toBe('trace-abc')
  })

  it('large tensor uses chunked framing', async () => {
    const handler: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (_sid, input) => input,
      onClose: async () => {},
    }

    try { await nodeB.libp2p.unhandle(KV_PROTOCOL) } catch {}
    await registerKVHandler(nodeB.libp2p, handler)

    const client = await KVSessionClient.open(nodeA.libp2p, nodeB.libp2p.peerId, 'sess-3', 2048)
    // 256 tokens × 128 dim × 4 bytes = 128KB — exceeds DEFAULT_CHUNK_SIZE
    const big = new Float32Array(256 * 128).fill(0.1)
    const out = await client.forward(big, 256, 128)
    expect(out.length).toBe(256 * 128)
    await client.close()
  })

  it('rollback resets handler state', async () => {
    let lastRollbackSession: string | undefined
    let lastRollbackNPast: number | undefined

    const handler: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (_sid, input) => input,
      onClose: async () => {},
      onRollback: async (sessionId, newNPast) => {
        lastRollbackSession = sessionId
        lastRollbackNPast = newNPast
      },
    }

    try { await nodeB.libp2p.unhandle(KV_PROTOCOL) } catch {}
    await registerKVHandler(nodeB.libp2p, handler)

    const client = await KVSessionClient.open(
      nodeA.libp2p, nodeB.libp2p.peerId, 'sess-rb', 128,
    )

    await client.rollback(5)

    expect(lastRollbackSession).toBe('sess-rb')
    expect(lastRollbackNPast).toBe(5)

    await client.close()
  })

  it('forwardAll returns nTokens × vocabSize floats', async () => {
    const vocabSize = 8
    const nTokens = 3
    const nEmbd = 4

    const handler: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (_sid, input) => input,
      onClose: async () => {},
      onForwardAll: async (_sid, _input, n) => {
        return new Float32Array(n * vocabSize).fill(0.42)
      },
    }

    try { await nodeB.libp2p.unhandle(KV_PROTOCOL) } catch {}
    await registerKVHandler(nodeB.libp2p, handler)

    const client = await KVSessionClient.open(nodeA.libp2p, nodeB.libp2p.peerId, 'sess-fa', 128)
    const input = new Float32Array(nTokens * nEmbd).fill(1.0)
    const out = await client.forwardAll(input, nTokens, nEmbd)

    expect(out.length).toBe(nTokens * vocabSize)
    for (let i = 0; i < out.length; i++) expect(out[i]).toBeCloseTo(0.42, 5)

    await client.close()
  })

  it('forwardAll returns STATUS_ERR when onForwardAll not implemented', async () => {
    const handler: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (_sid, input) => input,
      onClose: async () => {},
    }

    try { await nodeB.libp2p.unhandle(KV_PROTOCOL) } catch {}
    await registerKVHandler(nodeB.libp2p, handler)

    const client = await KVSessionClient.open(nodeA.libp2p, nodeB.libp2p.peerId, 'sess-fa-err', 128)
    const input = new Float32Array(4).fill(1.0)

    let threwError = false
    try {
      await client.forwardAll(input, 1, 4)
    } catch {
      threwError = true
    }
    expect(threwError).toBe(true)
  })
})
