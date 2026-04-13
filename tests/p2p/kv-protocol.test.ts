import { describe, it, expect, beforeAll, afterAll } from 'bun:test'
import { createOpenCoralNode, type OpenCoralNode } from '../../src/p2p/node'
import {
  KV_PROTOCOL,
  registerKVHandler,
  openRemoteSession,
  forwardRemote,
  closeRemoteSession,
  type KVSessionHandler,
} from '../../src/p2p/kv-protocol'

describe('KV session protocol', () => {
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
    expect(KV_PROTOCOL).toBe('/opencoral/kv/1.0.0')
  })

  it('open → forward → close lifecycle works', async () => {
    const sessions = new Map<string, Float32Array[]>()

    const handler: KVSessionHandler = {
      onOpen: async (sessionId, _maxSeqLen) => {
        sessions.set(sessionId, [])
        return { ok: true }
      },
      onForward: async (sessionId, input, _nTokens, _nEmbd) => {
        sessions.get(sessionId)!.push(input)
        // Echo the input as output (identity transform)
        return input
      },
      onClose: async (sessionId) => {
        sessions.delete(sessionId)
      },
    }

    await registerKVHandler(nodeB.libp2p, handler)

    const sessionId = 'test-session-1'
    await openRemoteSession(nodeA.libp2p, nodeB.libp2p.peerId, sessionId, 128)

    const input = new Float32Array(2 * 4).fill(0.7)
    const output = await forwardRemote(nodeA.libp2p, nodeB.libp2p.peerId, sessionId, input, 2, 4)

    expect(output).toBeInstanceOf(Float32Array)
    expect(output.length).toBe(2 * 4)
    for (let i = 0; i < input.length; i++) {
      expect(output[i]).toBeCloseTo(0.7, 5)
    }

    await closeRemoteSession(nodeA.libp2p, nodeB.libp2p.peerId, sessionId)
    expect(sessions.has(sessionId)).toBe(false)
  })

  it('forwardRemote on unknown session throws', async () => {
    const handler: KVSessionHandler = {
      onOpen: async () => ({ ok: true }),
      onForward: async (sessionId) => { throw new Error(`Unknown session: ${sessionId}`) },
      onClose: async () => {},
    }
    try { await nodeB.libp2p.unhandle(KV_PROTOCOL) } catch {}
    await registerKVHandler(nodeB.libp2p, handler)

    const input = new Float32Array(4)
    await expect(
      forwardRemote(nodeA.libp2p, nodeB.libp2p.peerId, 'no-such-session', input, 1, 4)
    ).rejects.toThrow()
  })
})
