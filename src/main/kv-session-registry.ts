import type { AsyncBlockRunner } from '../inference/native-worker'
import type { KVSessionHandler } from '../p2p/kv-protocol'

interface Entry {
  nativeSessionId: number
  lastUsed: number
}

/**
 * Tracks open KV sessions for one AsyncBlockRunner. Builds a KVSessionHandler
 * that delegates to the runner; evicts sessions that have been idle past
 * idleTimeoutMs. Lives in src/main/ because it's only used by the block host,
 * but has no dependency on Electron or libp2p.
 */
export class KVSessionRegistry {
  private readonly sessions = new Map<string, Entry>()
  private readonly cleanupTimer: ReturnType<typeof setInterval>

  constructor(
    private readonly runner: AsyncBlockRunner,
    private readonly idleTimeoutMs = 5 * 60 * 1000,
    cleanupIntervalMs = 60_000,
  ) {
    this.cleanupTimer = setInterval(() => this.cleanupStale(), cleanupIntervalMs)
  }

  async open(sessionId: string, maxSeqLen: number): Promise<void> {
    if (this.sessions.has(sessionId)) return
    const nativeSessionId = await this.runner.openSession(maxSeqLen)
    this.sessions.set(sessionId, { nativeSessionId, lastUsed: Date.now() })
  }

  async close(sessionId: string): Promise<void> {
    const entry = this.sessions.get(sessionId)
    if (!entry) return
    this.sessions.delete(sessionId)
    try {
      await this.runner.closeSession(entry.nativeSessionId)
    } catch {
      // close errors are best-effort; the session is already gone from the registry
    }
  }

  async dispose(): Promise<void> {
    clearInterval(this.cleanupTimer)
    const entries = Array.from(this.sessions.values())
    this.sessions.clear()
    await Promise.allSettled(entries.map(e => this.runner.closeSession(e.nativeSessionId)))
  }

  buildHandler(): KVSessionHandler {
    // onForwardAll is intentionally omitted. The native projection path
    // (llama_project_hidden_to_logits) is blocked on a pre-existing patch bug —
    // the projection-only graph is not reserved by the ggml scheduler, so all
    // non-shim contexts crash. Until the patch is fixed, MSG_FORWARD_ALL over the
    // wire returns STATUS_ERR "onForwardAll not implemented" and callers (KVChain)
    // surface that as a thrown error on the client side.
    return {
      onOpen: async (sessionId, maxSeqLen) => {
        await this.open(sessionId, maxSeqLen)
        return { ok: true }
      },
      onForward: async (sessionId, input, nTokens) => {
        const entry = this.touch(sessionId)
        return this.runner.sessionForward(entry.nativeSessionId, input, nTokens)
      },
      onClose: async (sessionId) => {
        await this.close(sessionId)
      },
      onRollback: async (sessionId, newNPast) => {
        const entry = this.touch(sessionId)
        await this.runner.sessionRollback(entry.nativeSessionId, newNPast)
      },
    }
  }

  private touch(sessionId: string): Entry {
    const entry = this.sessions.get(sessionId)
    if (!entry) throw new Error(`KV session not found: ${sessionId}`)
    entry.lastUsed = Date.now()
    return entry
  }

  private cleanupStale(): void {
    const now = Date.now()
    for (const [id, entry] of this.sessions) {
      if (now - entry.lastUsed > this.idleTimeoutMs) {
        this.runner.closeSession(entry.nativeSessionId).catch(() => {})
        this.sessions.delete(id)
        console.log(`[OpenCoral KV] Cleaned up stale session ${id}`)
      }
    }
  }
}
