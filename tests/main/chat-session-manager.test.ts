// tests/main/chat-session-manager.test.ts
import { describe, it, expect, beforeEach, afterEach } from 'bun:test'
import { promises as fs } from 'fs'
import { join } from 'path'
import { tmpdir } from 'os'
import { SessionStore } from '../../src/main/session-store'
import { ChatSessionManager, type ChatSessionManagerDeps } from '../../src/main/chat-session-manager'
import type { AsyncBlockRunner } from '../../src/inference/native-worker'
import type { NativeTokenizer, ChatTurn } from '../../src/inference/native-tokenizer'

let dir: string

beforeEach(async () => { dir = await fs.mkdtemp(join(tmpdir(), 'opencoral-mgr-')) })
afterEach(async () => { await fs.rm(dir, { recursive: true, force: true }) })

function sharpLogits(vocabSize: number, nTokens: number, id: number): Float32Array {
  const f = new Float32Array(nTokens * vocabSize)
  for (let t = 0; t < nTokens; t++) f[t * vocabSize + id] = 100
  return f
}

function makeMockRunner(vocabSize: number, eos: number) {
  const calls = { openSession: 0, closeSession: 0, decodeAll: 0, decode: 0, rollback: 0, embed: 0 }
  const runner = {
    vocabSize,
    hiddenSize: 16,
    blockStart: 0, blockEnd: -1,
    openSession: async (_n: number) => { calls.openSession++; return 1 },
    closeSession: async (_id: number) => { calls.closeSession++ },
    sessionDecodeLogitsAll: async (_sid: number, ids: Int32Array) => {
      calls.decodeAll++
      return sharpLogits(vocabSize, ids.length, eos)
    },
    sessionDecodeLogits: async (_sid: number, ids: Int32Array) => {
      calls.decode++
      return sharpLogits(vocabSize, ids.length, eos)
    },
    sessionRollback: async (_sid: number, _n: number) => { calls.rollback++ },
    // KVChain (remote-step path) calls embedTokens via the embedder adapter.
    embedTokens: async (ids: Int32Array) => {
      calls.embed++
      return new Float32Array(ids.length * 16)
    },
  }
  return { runner: runner as unknown as AsyncBlockRunner, calls }
}

function makeFakeKvClient(vocabSize: number, eos: number) {
  // Minimal KVSessionClient shape used by KVChain. forward returns hidden states
  // unchanged (chain is single-step in tests). forwardAll returns sharp-EOS logits.
  return {
    forward: async (h: Float32Array, _nTokens: number, _nEmbd: number) => h,
    forwardAll: async (_h: Float32Array, nTokens: number, _nEmbd: number) =>
      sharpLogits(vocabSize, nTokens, eos),
    rollback: async (_n: number) => {},
    close: async () => {},
  }
}

function makeMockTokenizer(vocabSize: number, eos: number) {
  const tok = {
    vocabSize, bosTokenId: 1, eosTokenId: eos, endOfTurnTokenId: undefined,
    encode: async (s: string) => new Int32Array(s.split('').map(c => c.charCodeAt(0))),
    encodeChat: async (s: string) => new Int32Array([1, ...s.split('').map(c => c.charCodeAt(0))]),
    encodeChatMulti: async (turns: ChatTurn[]) => {
      const out: number[] = [1]
      for (const t of turns) {
        out.push(t.role === 'user' ? 100 : t.role === 'assistant' ? 101 : 102)
        for (const c of t.content) out.push(c.charCodeAt(0))
        out.push(99)
      }
      out.push(101)
      return new Int32Array(out)
    },
    decode: async (ids: number[]) => ids.map(id => String.fromCharCode(id)).join(''),
    decodeToken: async (id: number) => String.fromCharCode(id),
  }
  return tok as unknown as NativeTokenizer
}

function makeDeps(extra: Partial<ChatSessionManagerDeps>): ChatSessionManagerDeps {
  return {
    store: extra.store!,
    getRunner: extra.getRunner ?? (() => null),
    getTokenizer: extra.getTokenizer ?? (() => null),
    getPlanner: extra.getPlanner ?? (() => null),
    emitPhase: extra.emitPhase ?? (() => {}),
    emitInvalidation: extra.emitInvalidation ?? (() => {}),
  }
}

describe('ChatSessionManager.openTurn — cold start', () => {
  it('persists user message before running inference', async () => {
    const store = new SessionStore(dir)
    await store.init()

    const session = {
      schemaVersion: 1 as const, id: 's1', title: 'New chat',
      createdAt: 1000, updatedAt: 1000, messages: [],
    }
    await store.save(session); await store.flush()

    const { runner } = makeMockRunner(256, 0)
    const tokenizer = makeMockTokenizer(256, 0)

    const mgr = new ChatSessionManager(makeDeps({
      store,
      getRunner: () => runner,
      getTokenizer: () => tokenizer,
      getPlanner: () => ({
        plan: async () => ({
          localSteps: [{ blockStart: 0, blockEnd: 31 }],
          remoteSteps: [],
        }),
        openRemoteKv: async () => [],
      }),
    }))

    await mgr.openTurn('s1', 'Hello world', 8)

    const after = await store.load('s1')
    expect(after?.messages[0]?.role).toBe('user')
    expect(after?.messages[0]?.text).toBe('Hello world')
  })
})

describe('ChatSessionManager — delta tokenization', () => {
  it('does not roll back KV when a new turn purely extends the prior history', async () => {
    const store = new SessionStore(dir)
    await store.init()
    await store.save({
      schemaVersion: 1, id: 's2', title: 'New chat',
      createdAt: 1, updatedAt: 1, messages: [],
    })
    await store.flush()

    const { runner, calls } = makeMockRunner(256, 0)
    const tokenizer = makeMockTokenizer(256, 0)

    const mgr = new ChatSessionManager(makeDeps({
      store, getRunner: () => runner, getTokenizer: () => tokenizer,
      getPlanner: () => ({
        plan: async () => ({ localSteps: [{ blockStart: 0, blockEnd: 0 }], remoteSteps: [] }),
        openRemoteKv: async () => [],
      }),
    }))

    await mgr.openTurn('s2', 'first', 4)
    await mgr.openTurn('s2', 'second', 4)

    expect(calls.rollback).toBe(0)
    expect(calls.openSession).toBe(1)
  })

  it('tears down KV when a different session is opened', async () => {
    const store = new SessionStore(dir)
    await store.init()
    for (const id of ['a', 'b']) {
      await store.save({
        schemaVersion: 1, id, title: 'New chat',
        createdAt: 1, updatedAt: 1, messages: [],
      })
      await store.flush()
    }

    const { runner, calls } = makeMockRunner(256, 0)
    const tokenizer = makeMockTokenizer(256, 0)
    const mgr = new ChatSessionManager(makeDeps({
      store, getRunner: () => runner, getTokenizer: () => tokenizer,
      getPlanner: () => ({
        plan: async () => ({ localSteps: [{ blockStart: 0, blockEnd: 0 }], remoteSteps: [] }),
        openRemoteKv: async () => [],
      }),
    }))

    await mgr.openTurn('a', 'hi from a', 4)
    expect(calls.openSession).toBe(1)
    expect(calls.closeSession).toBe(0)

    await mgr.openTurn('b', 'hi from b', 4)
    expect(calls.openSession).toBe(2)
    expect(calls.closeSession).toBe(1)
  })
})

describe('ChatSessionManager — peer-drop invalidation', () => {
  it('invalidates active context when a peer in the chain disconnects', async () => {
    const store = new SessionStore(dir)
    await store.init()
    await store.save({
      schemaVersion: 1, id: 'p1', title: 'New chat',
      createdAt: 1, updatedAt: 1, messages: [],
    })
    await store.flush()

    const { runner } = makeMockRunner(256, 0)
    const tokenizer = makeMockTokenizer(256, 0)

    let invalidations: Array<{ sessionId: string; reason: string }> = []
    const closedRemotes: string[] = []

    const mgr = new ChatSessionManager(makeDeps({
      store, getRunner: () => runner, getTokenizer: () => tokenizer,
      getPlanner: () => ({
        plan: async () => ({
          localSteps: [{ blockStart: 0, blockEnd: 15 }],
          remoteSteps: [{ peerId: 'peerX', blockStart: 16, blockEnd: 31 }],
        }),
        openRemoteKv: async () => {
          const client = makeFakeKvClient(256, 0)
          return [{
            peerId: 'peerX',
            client: client as unknown as import('../../src/p2p/kv-protocol').KVSessionClient,
            close: async () => { closedRemotes.push('peerX'); await client.close() },
          }]
        },
      }),
      emitInvalidation: (e) => { invalidations.push(e) },
    }))

    await mgr.openTurn('p1', 'hi', 4)
    expect(mgr.hasActive()).toBe(true)

    mgr.notifyPeerDisconnected('peerX')
    await new Promise(r => setTimeout(r, 10))

    expect(invalidations).toEqual([{ sessionId: 'p1', reason: 'peer-drop' }])
    expect(closedRemotes).toEqual(['peerX'])
    expect(mgr.hasActive()).toBe(false)
  })

  it('ignores peer-disconnect for peers not in active chain', async () => {
    const store = new SessionStore(dir)
    await store.init()
    await store.save({
      schemaVersion: 1, id: 'p2', title: 'New chat',
      createdAt: 1, updatedAt: 1, messages: [],
    })
    await store.flush()

    const { runner } = makeMockRunner(256, 0)
    const tokenizer = makeMockTokenizer(256, 0)
    const mgr = new ChatSessionManager(makeDeps({
      store, getRunner: () => runner, getTokenizer: () => tokenizer,
      getPlanner: () => ({
        plan: async () => ({ localSteps: [{ blockStart: 0, blockEnd: 31 }], remoteSteps: [] }),
        openRemoteKv: async () => [],
      }),
    }))
    await mgr.openTurn('p2', 'hi', 4)
    mgr.notifyPeerDisconnected('peerY')
    expect(mgr.hasActive()).toBe(true)
  })
})

describe('ChatSessionManager — concurrency and phases', () => {
  it('rejects a second openTurn on the same session while one is in flight', async () => {
    const store = new SessionStore(dir)
    await store.init()
    await store.save({
      schemaVersion: 1, id: 'c1', title: 'New chat',
      createdAt: 1, updatedAt: 1, messages: [],
    })
    await store.flush()

    const { runner } = makeMockRunner(256, 0)
    const tokenizer = makeMockTokenizer(256, 0)
    const mgr = new ChatSessionManager(makeDeps({
      store, getRunner: () => runner, getTokenizer: () => tokenizer,
      getPlanner: () => ({
        plan: async () => ({ localSteps: [{ blockStart: 0, blockEnd: 0 }], remoteSteps: [] }),
        openRemoteKv: async () => [],
      }),
    }))

    const p1 = mgr.openTurn('c1', 'first', 4)
    let secondError: unknown = null
    try { await mgr.openTurn('c1', 'second', 4) } catch (e) { secondError = e }
    expect((secondError as Error)?.message).toMatch(/in-flight/)
    await p1
  })

  it('emits planning → ready when no remote steps exist', async () => {
    const store = new SessionStore(dir)
    await store.init()
    await store.save({
      schemaVersion: 1, id: 'ph1', title: 'New chat',
      createdAt: 1, updatedAt: 1, messages: [],
    })
    await store.flush()

    const phases: string[] = []
    const { runner } = makeMockRunner(256, 0)
    const tokenizer = makeMockTokenizer(256, 0)
    const mgr = new ChatSessionManager(makeDeps({
      store, getRunner: () => runner, getTokenizer: () => tokenizer,
      getPlanner: () => ({
        plan: async () => ({ localSteps: [{ blockStart: 0, blockEnd: 0 }], remoteSteps: [] }),
        openRemoteKv: async () => [],
      }),
      emitPhase: (e) => { phases.push(e.phase) },
    }))
    await mgr.openTurn('ph1', 'hi', 4)
    expect(phases).toEqual(['planning', 'ready'])
  })
})

describe('ChatSessionManager — full-chain KV (Phase 6)', () => {
  it('uses KVChain backend when remote steps exist (calls embedder + remote client)', async () => {
    const store = new SessionStore(dir)
    await store.init()
    await store.save({
      schemaVersion: 1, id: 'r1', title: 'New chat',
      createdAt: 1, updatedAt: 1, messages: [],
    })
    await store.flush()

    const { runner, calls } = makeMockRunner(256, 0)
    const tokenizer = makeMockTokenizer(256, 0)

    const opened: string[] = []
    const closed: string[] = []
    const fakeClient = makeFakeKvClient(256, 0)

    const mgr = new ChatSessionManager(makeDeps({
      store, getRunner: () => runner, getTokenizer: () => tokenizer,
      getPlanner: () => ({
        plan: async () => ({
          localSteps: [{ blockStart: 0, blockEnd: 15 }],
          remoteSteps: [{ peerId: 'p1', blockStart: 16, blockEnd: 31 }],
        }),
        openRemoteKv: async (steps) => {
          for (const s of steps) opened.push(s.peerId)
          return steps.map(s => ({
            peerId: s.peerId,
            client: fakeClient as unknown as import('../../src/p2p/kv-protocol').KVSessionClient,
            close: async () => { closed.push(s.peerId) },
          }))
        },
      }),
    }))

    await mgr.openTurn('r1', 'hi', 4)
    expect(opened).toEqual(['p1'])
    // KVChain path uses runner.embedTokens (NOT runner.sessionDecodeLogitsAll)
    // because embedding is local-stateless; the chain handles KV state.
    expect(calls.embed).toBeGreaterThan(0)

    await mgr.invalidateActive('peer-drop')
    expect(closed).toEqual(['p1'])
  })
})
