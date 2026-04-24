import { describe, it, expect } from 'bun:test'
import { KVSessionRegistry } from '../../src/main/kv-session-registry'
import type { AsyncBlockRunner } from '../../src/inference/native-worker'

interface Call {
  method: string
  args: unknown[]
}

function makeMockRunner(opts: {
  openReturns?: number
  sessionForwardImpl?: (nativeId: number, input: Float32Array, nTokens: number) => Promise<Float32Array>
  projectImpl?: (input: Float32Array, nTokens: number) => Promise<Float32Array>
  closeImpl?: (nativeId: number) => Promise<void>
  rollbackImpl?: (nativeId: number, newNPast: number) => Promise<void>
} = {}) {
  const calls: Call[] = []
  let nextNativeId = 100
  const runner = {
    openSession: async (maxSeqLen: number) => {
      calls.push({ method: 'openSession', args: [maxSeqLen] })
      return opts.openReturns ?? nextNativeId++
    },
    sessionForward: async (nativeId: number, input: Float32Array, nTokens: number) => {
      calls.push({ method: 'sessionForward', args: [nativeId, input, nTokens] })
      return opts.sessionForwardImpl
        ? opts.sessionForwardImpl(nativeId, input, nTokens)
        : new Float32Array(nTokens * 4)
    },
    projectToLogitsAll: async (input: Float32Array, nTokens: number) => {
      calls.push({ method: 'projectToLogitsAll', args: [input, nTokens] })
      return opts.projectImpl
        ? opts.projectImpl(input, nTokens)
        : new Float32Array(nTokens * 8)
    },
    closeSession: async (nativeId: number) => {
      calls.push({ method: 'closeSession', args: [nativeId] })
      if (opts.closeImpl) return opts.closeImpl(nativeId)
    },
    sessionRollback: async (nativeId: number, newNPast: number) => {
      calls.push({ method: 'sessionRollback', args: [nativeId, newNPast] })
      if (opts.rollbackImpl) return opts.rollbackImpl(nativeId, newNPast)
    },
  }
  return { runner: runner as unknown as AsyncBlockRunner, calls }
}

describe('KVSessionRegistry', () => {
  it('open is idempotent for same sessionId', async () => {
    const { runner, calls } = makeMockRunner()
    const reg = new KVSessionRegistry(runner)
    await reg.open('s1', 128)
    await reg.open('s1', 256)  // second call no-ops
    const opens = calls.filter(c => c.method === 'openSession')
    expect(opens.length).toBe(1)
    expect(opens[0].args).toEqual([128])
    await reg.dispose()
  })

  it('close removes session and calls runner.closeSession', async () => {
    const { runner, calls } = makeMockRunner({ openReturns: 42 })
    const reg = new KVSessionRegistry(runner)
    await reg.open('s1', 64)
    await reg.close('s1')
    const closes = calls.filter(c => c.method === 'closeSession')
    expect(closes.length).toBe(1)
    expect(closes[0].args).toEqual([42])
    // subsequent forward on the closed session should throw
    await expect(reg.buildHandler().onForward('s1', new Float32Array(4), 1, 4)).rejects.toThrow()
    await reg.dispose()
  })

  it('close on unknown session is a no-op', async () => {
    const { runner, calls } = makeMockRunner()
    const reg = new KVSessionRegistry(runner)
    await reg.close('never-opened')
    expect(calls.filter(c => c.method === 'closeSession').length).toBe(0)
    await reg.dispose()
  })

  it('stale sessions are cleaned up after idle timeout', async () => {
    const { runner, calls } = makeMockRunner({ openReturns: 7 })
    const reg = new KVSessionRegistry(runner, 30, 10)  // 30ms idle, 10ms sweep
    await reg.open('s1', 32)
    await new Promise(r => setTimeout(r, 80))
    const closes = calls.filter(c => c.method === 'closeSession')
    expect(closes.length).toBe(1)
    expect(closes[0].args).toEqual([7])
    await reg.dispose()
  })

  it('dispose stops timer and closes outstanding sessions', async () => {
    const { runner, calls } = makeMockRunner()
    const reg = new KVSessionRegistry(runner, 10_000, 10_000)
    await reg.open('s1', 32)
    await reg.open('s2', 32)
    await reg.dispose()
    const closes = calls.filter(c => c.method === 'closeSession')
    expect(closes.length).toBe(2)
  })

  it('buildHandler.onOpen delegates to open and returns {ok: true}', async () => {
    const { runner, calls } = makeMockRunner()
    const reg = new KVSessionRegistry(runner)
    const h = reg.buildHandler()
    const result = await h.onOpen('s1', 64)
    expect(result).toEqual({ ok: true })
    expect(calls.filter(c => c.method === 'openSession').length).toBe(1)
    await reg.dispose()
  })

  it('buildHandler.onForward throws on unknown session', async () => {
    const { runner } = makeMockRunner()
    const reg = new KVSessionRegistry(runner)
    const h = reg.buildHandler()
    await expect(h.onForward('bogus', new Float32Array(4), 1, 4)).rejects.toThrow('KV session not found')
    await reg.dispose()
  })

  it('buildHandler.onForward calls runner.sessionForward with native id', async () => {
    const { runner, calls } = makeMockRunner({ openReturns: 55 })
    const reg = new KVSessionRegistry(runner)
    const h = reg.buildHandler()
    await h.onOpen('s1', 64)
    const input = new Float32Array(8).fill(0.3)
    await h.onForward('s1', input, 2, 4)
    const fwd = calls.filter(c => c.method === 'sessionForward')
    expect(fwd.length).toBe(1)
    expect(fwd[0].args[0]).toBe(55)
    expect(fwd[0].args[2]).toBe(2)
    await reg.dispose()
  })

  it('buildHandler does NOT expose onForwardAll (blocked on native patch bug)', async () => {
    const { runner } = makeMockRunner()
    const reg = new KVSessionRegistry(runner)
    const h = reg.buildHandler()
    expect(h.onForwardAll).toBeUndefined()
    await reg.dispose()
  })

  it('buildHandler.onRollback calls runner.sessionRollback with native id', async () => {
    const { runner, calls } = makeMockRunner({ openReturns: 99 })
    const reg = new KVSessionRegistry(runner)
    const h = reg.buildHandler()
    await h.onOpen('s1', 64)
    await h.onRollback!('s1', 5)
    const rb = calls.filter(c => c.method === 'sessionRollback')
    expect(rb.length).toBe(1)
    expect(rb[0].args).toEqual([99, 5])
    await reg.dispose()
  })

  it('buildHandler.onClose removes session', async () => {
    const { runner, calls } = makeMockRunner({ openReturns: 13 })
    const reg = new KVSessionRegistry(runner)
    const h = reg.buildHandler()
    await h.onOpen('s1', 64)
    await h.onClose('s1')
    expect(calls.filter(c => c.method === 'closeSession').length).toBe(1)
    // subsequent onForward should throw
    await expect(h.onForward('s1', new Float32Array(4), 1, 4)).rejects.toThrow('KV session not found')
    await reg.dispose()
  })

  it('active session is not evicted by cleanup', async () => {
    const { runner, calls } = makeMockRunner()
    const reg = new KVSessionRegistry(runner, 50, 10)
    const h = reg.buildHandler()
    await h.onOpen('s1', 64)
    // Touch every 20ms for 100ms (keeps the session active)
    for (let i = 0; i < 5; i++) {
      await h.onForward('s1', new Float32Array(4), 1, 4)
      await new Promise(r => setTimeout(r, 20))
    }
    expect(calls.filter(c => c.method === 'closeSession').length).toBe(0)
    await reg.dispose()
  })
})
