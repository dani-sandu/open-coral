// src/main/chat-session-manager.ts

import { randomUUID } from 'crypto'
import type { AsyncBlockRunner } from '../inference/native-worker'
import type { NativeTokenizer, ChatTurn } from '../inference/native-tokenizer'
import type { SessionStore, PersistedChatSession, PersistedChatMessage, InferenceTrace } from './session-store'
import type { SessionSummary } from './session-store'
import type { KVSessionClient } from '../p2p/kv-protocol'
import { LocalVerificationBackend, SpeculativeSession, DEFAULT_SPEC_CONFIG } from '../inference/speculative-session'
import type { VerificationBackend } from '../inference/speculative-session'

export type SessionPhase = 'planning' | 'opening-remote-kv' | 'prefilling' | 'ready' | 'error'

export interface SessionPhaseEvent {
  sessionId: string
  phase: SessionPhase
  prefilledTokens?: number
  totalTokens?: number
  error?: string
}

export interface InvalidationEvent {
  sessionId: string
  reason: 'peer-drop' | 'model-change'
}

export interface RemoteKvHandle {
  peerId: string
  /** Underlying KVSessionClient — used by KVChain to chain forwards. */
  client: KVSessionClient
  close(): Promise<void>
}

export interface ChainPlanner {
  plan(): Promise<{
    localSteps: { blockStart: number; blockEnd: number }[]
    remoteSteps: { peerId: string; blockStart: number; blockEnd: number; multiaddr?: string }[]
  }>
  openRemoteKv(remoteSteps: { peerId: string; multiaddr?: string }[], maxSeqLen: number): Promise<RemoteKvHandle[]>
}

export interface TurnResult {
  generatedText: string
  trace: InferenceTrace
}

export interface ChatSessionManagerDeps {
  store: SessionStore
  getRunner: () => AsyncBlockRunner | null
  getTokenizer: () => NativeTokenizer | null
  getPlanner: () => ChainPlanner | null
  emitPhase: (e: SessionPhaseEvent) => void
  emitInvalidation: (e: InvalidationEvent) => void
}

interface ActiveTurnContext {
  sessionId: string
  kvSessionId: number
  remoteKvHandles: RemoteKvHandle[]
  chain: { localSteps: { blockStart: number; blockEnd: number }[]; remoteSteps: { peerId: string; blockStart: number; blockEnd: number; multiaddr?: string }[] }
  nPast: number
  tokenizedHistory: Int32Array
  peerIdsInUse: Set<string>
  /** Long-lived backend reused across turns. KVChain when remote steps exist; LocalVerificationBackend otherwise. */
  backend: VerificationBackend
}

const KV_RESERVE_FOR_GENERATION = 2048
/** Max tokens per prefill batch. Stays under llama.cpp's default n_ubatch (512). */
const PREFILL_CHUNK_SIZE = 256

export class ChatSessionManager {
  private readonly deps: ChatSessionManagerDeps
  private active: ActiveTurnContext | null = null
  private inflight: Map<string, Promise<unknown>> = new Map()

  constructor(deps: ChatSessionManagerDeps) {
    this.deps = deps
  }

  hasActive(): boolean { return this.active !== null }

  /** Returns the id of the currently warm session, or null. */
  activeSessionId(): string | null { return this.active?.sessionId ?? null }

  /** Tear down the active KV without emitting an invalidation event. Used when
   *  the caller will inform the renderer through a different channel (e.g. a
   *  session-deleted broadcast) and a "context lost" toast would be misleading. */
  async tearDownActive(): Promise<void> {
    if (!this.active) return
    await this.teardown(this.active)
    this.active = null
  }

  /** Tear down + emit invalidation so the renderer can show a recovery toast. */
  async invalidateActive(reason: 'peer-drop' | 'model-change'): Promise<void> {
    if (!this.active) return
    const sid = this.active.sessionId
    await this.teardown(this.active)
    this.active = null
    this.deps.emitInvalidation({ sessionId: sid, reason })
  }

  async shutdown(): Promise<void> {
    if (this.active) {
      await this.teardown(this.active)
      this.active = null
    }
  }

  async openTurn(sessionId: string, userText: string, maxTokens: number): Promise<TurnResult> {
    const existing = this.inflight.get(sessionId)
    if (existing) {
      throw new Error(`Session ${sessionId} already has an in-flight turn`)
    }

    const promise = this._runTurn(sessionId, userText, maxTokens).finally(() => {
      this.inflight.delete(sessionId)
    })
    this.inflight.set(sessionId, promise)
    return promise as Promise<TurnResult>
  }

  private async _runTurn(sessionId: string, userText: string, maxTokens: number): Promise<TurnResult> {
    const runner = this.deps.getRunner()
    const tokenizer = this.deps.getTokenizer()
    if (!runner) throw new Error('No model runner available')
    if (!tokenizer) throw new Error('No tokenizer available')

    if (this.active && this.active.sessionId !== sessionId) {
      await this.teardown(this.active)
      this.active = null
    }

    if (!this.active) {
      this.active = await this.coldStart(sessionId, runner, tokenizer)
    }

    const active = this.active

    const session = await this.deps.store.load(sessionId)
    if (!session) throw new Error(`Session ${sessionId} not found`)

    const userMsg: PersistedChatMessage = {
      id: `${Date.now()}-user`,
      role: 'user',
      text: userText,
      timestamp: Date.now(),
    }
    session.messages.push(userMsg)
    session.updatedAt = Date.now()
    // Auto-title from the first user message. After push() above, length is 1 on
    // the very first turn — the only point where we want to capture the title.
    if (session.messages.length === 1) {
      session.title = userText.length > 40 ? userText.slice(0, 40) + '…' : userText
    }
    await this.deps.store.save(session)

    const turns: ChatTurn[] = session.messages
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .map(m => ({ role: m.role, content: m.text }))

    const afterTokens = await tokenizer.encodeChatMulti(turns)
    const commonLen = longestCommonPrefix(active.tokenizedHistory, afterTokens)

    if (commonLen < active.nPast) {
      await runner.sessionRollback(active.kvSessionId, commonLen)
      active.nPast = commonLen
    }

    const newTokens = afterTokens.slice(commonLen)
    if (newTokens.length === 0) {
      throw new Error('No new tokens to feed — duplicate turn?')
    }

    const backend = active.backend
    backend.nPast = active.nPast

    // Pre-feed all but the last chunk so the batch handed to SpeculativeSession
    // stays under llama.cpp's n_ubatch. This matters when chat templates aren't
    // prefix-stable and the delta covers more than just the user's message.
    let prefedTokens = newTokens
    while (prefedTokens.length > PREFILL_CHUNK_SIZE) {
      const chunk = prefedTokens.subarray(0, PREFILL_CHUNK_SIZE)
      await backend.forwardAll(chunk)
      prefedTokens = prefedTokens.subarray(PREFILL_CHUNK_SIZE)
    }

    const spec = new SpeculativeSession(
      backend,
      tokenizer.eosTokenId,
      tokenizer.endOfTurnTokenId,
      DEFAULT_SPEC_CONFIG,
    )

    const t0 = Date.now()
    const gen = await spec.generate(prefedTokens, maxTokens)
    const inferenceMs = Date.now() - t0

    active.nPast = backend.nPast
    active.tokenizedHistory = concatInt32(afterTokens, new Int32Array(gen.tokenIds))

    const generatedText = await tokenizer.decode(gen.tokenIds)
    const trace: InferenceTrace = {
      prompt: userText,
      generatedText,
      generatedTokens: gen.tokenIds.length,
      nEmbd: runner.hiddenSize,
      chainSteps: [
        ...active.chain.localSteps.map(s => ({ peerId: 'local', ...s, durationMs: inferenceMs })),
        ...active.chain.remoteSteps.map(s => ({ peerId: s.peerId, blockStart: s.blockStart, blockEnd: s.blockEnd, durationMs: 0 })),
      ],
      totalDurationMs: inferenceMs,
      specDraftTokens: gen.specDraftTokens > 0 ? gen.specDraftTokens : undefined,
      specAcceptedTokens: gen.specDraftTokens > 0 ? gen.specAcceptedTokens : undefined,
      specAcceptanceRate: gen.specDraftTokens > 0 ? gen.specAcceptedTokens / gen.specDraftTokens : undefined,
    }

    session.messages.push({
      id: `${Date.now()}-assistant`,
      role: 'assistant',
      text: generatedText,
      result: trace,
      timestamp: Date.now(),
    })
    session.updatedAt = Date.now()
    await this.deps.store.save(session)

    return { generatedText, trace }
  }

  private async coldStart(
    sessionId: string,
    runner: AsyncBlockRunner,
    tokenizer: NativeTokenizer,
  ): Promise<ActiveTurnContext> {
    const planner = this.deps.getPlanner()
    if (!planner) throw new Error('No chain planner available')

    this.deps.emitPhase({ sessionId, phase: 'planning' })
    const chain = await planner.plan()

    const session = await this.deps.store.load(sessionId)
    if (!session) throw new Error(`Session ${sessionId} not found`)

    const existingTurns: ChatTurn[] = session.messages
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .map(m => ({ role: m.role, content: m.text }))

    let priorTokens: Int32Array
    if (existingTurns.length === 0) {
      priorTokens = new Int32Array(0)
    } else {
      priorTokens = await tokenizer.encodeChatMulti(existingTurns)
    }

    const maxSeqLen = priorTokens.length + KV_RESERVE_FOR_GENERATION
    const kvSessionId = await runner.openSession(maxSeqLen)

    let remoteKvHandles: RemoteKvHandle[] = []
    if (chain.remoteSteps.length > 0) {
      this.deps.emitPhase({ sessionId, phase: 'opening-remote-kv' })
      try {
        remoteKvHandles = await planner.openRemoteKv(chain.remoteSteps, maxSeqLen)
      } catch (err) {
        await runner.closeSession(kvSessionId).catch(() => {})
        this.deps.emitPhase({ sessionId, phase: 'error', error: (err as Error).message })
        throw err
      }
    }

    const backend = await buildBackend(runner, kvSessionId, remoteKvHandles, tokenizer.vocabSize)

    if (priorTokens.length > 0) {
      this.deps.emitPhase({
        sessionId, phase: 'prefilling',
        prefilledTokens: 0, totalTokens: priorTokens.length,
      })
      try {
        // Chunk to stay under llama.cpp's n_ubatch (typically 512). Sending the
        // whole history in one shot fails with `llama_decode rc=-1` for any
        // history > n_ubatch tokens. Smaller chunks also let us report real
        // progress to the UI as the prefill advances.
        for (let i = 0; i < priorTokens.length; i += PREFILL_CHUNK_SIZE) {
          const end = Math.min(i + PREFILL_CHUNK_SIZE, priorTokens.length)
          const chunk = priorTokens.subarray(i, end)
          await backend.forwardAll(chunk)
          this.deps.emitPhase({
            sessionId, phase: 'prefilling',
            prefilledTokens: end, totalTokens: priorTokens.length,
          })
        }
      } catch (err) {
        // Roll back local + remote KV before propagating so the manager doesn't
        // leak handles into a never-installed ActiveTurnContext.
        for (const h of remoteKvHandles) await h.close().catch(() => {})
        await runner.closeSession(kvSessionId).catch(() => {})
        this.deps.emitPhase({ sessionId, phase: 'error', error: (err as Error).message })
        throw err
      }
    }

    this.deps.emitPhase({ sessionId, phase: 'ready' })

    return {
      sessionId,
      kvSessionId,
      remoteKvHandles,
      chain,
      nPast: priorTokens.length,
      tokenizedHistory: priorTokens,
      peerIdsInUse: new Set(chain.remoteSteps.map(s => s.peerId)),
      backend,
    }
  }

  private async teardown(ctx: ActiveTurnContext): Promise<void> {
    for (const h of ctx.remoteKvHandles) {
      await h.close().catch(err => console.warn(`[ChatSessionManager] remote KV close failed: ${(err as Error).message}`))
    }
    const runner = this.deps.getRunner()
    if (runner) {
      await runner.closeSession(ctx.kvSessionId).catch(err => console.warn(`[ChatSessionManager] local KV close failed: ${(err as Error).message}`))
    }
  }

  notifyPeerDisconnected(peerId: string): void {
    if (this.active && this.active.peerIdsInUse.has(peerId)) {
      this.invalidateActive('peer-drop').catch(() => {})
    }
  }
}

async function buildBackend(
  runner: AsyncBlockRunner,
  kvSessionId: number,
  remoteKvHandles: RemoteKvHandle[],
  vocabSize: number,
): Promise<VerificationBackend> {
  if (remoteKvHandles.length === 0) {
    return new LocalVerificationBackend(runner, kvSessionId)
  }
  // Stateless local embedder. Lazy-import KVChain so test mocks for the manager
  // don't have to satisfy KVChain's own dependencies.
  const embedder = {
    nEmbd: runner.hiddenSize,
    embed: (ids: Int32Array) => runner.embedTokens(ids),
  }
  const { KVChain } = await import('../p2p/kv-chain')
  return new KVChain(embedder, remoteKvHandles.map(h => h.client), vocabSize)
}

function longestCommonPrefix(a: Int32Array, b: Int32Array): number {
  const len = Math.min(a.length, b.length)
  for (let i = 0; i < len; i++) {
    if (a[i] !== b[i]) return i
  }
  return len
}

function concatInt32(a: Int32Array, b: Int32Array): Int32Array {
  const out = new Int32Array(a.length + b.length)
  out.set(a, 0); out.set(b, a.length)
  return out
}

// ── ChainPlanner adapter ─────────────────────────────────────────────────────
// Wraps the existing SequenceManager and KVSessionClient so the manager core
// stays free of libp2p / SequenceManager internals (easier to mock in tests).

import type { SequenceManager } from '../inference/sequence-manager'
import type { OpenCoralNode } from '../p2p/node'
import { KVSessionClient } from '../p2p/kv-protocol'
import { peerIdFromString } from '@libp2p/peer-id'

export interface ChainPlannerAdapterOptions {
  node: OpenCoralNode
  manager: SequenceManager
}

export class SequenceManagerChainPlanner implements ChainPlanner {
  private readonly node: OpenCoralNode
  private readonly mgr: SequenceManager

  constructor(opts: ChainPlannerAdapterOptions) {
    this.node = opts.node
    this.mgr = opts.manager
  }

  async plan(): Promise<{
    localSteps: { blockStart: number; blockEnd: number }[]
    remoteSteps: { peerId: string; blockStart: number; blockEnd: number; multiaddr?: string }[]
  }> {
    const steps = await this.mgr.planChainWithCandidates()
    const localSteps: { blockStart: number; blockEnd: number }[] = []
    const remoteSteps: { peerId: string; blockStart: number; blockEnd: number; multiaddr?: string }[] = []
    for (const s of steps) {
      const top = s.candidates[0]
      if (top.peerId === 'local') {
        localSteps.push({ blockStart: s.blockStart, blockEnd: s.blockEnd })
      } else {
        remoteSteps.push({ peerId: top.peerId, blockStart: s.blockStart, blockEnd: s.blockEnd, multiaddr: top.multiaddr })
      }
    }
    return { localSteps, remoteSteps }
  }

  async openRemoteKv(
    remoteSteps: { peerId: string; multiaddr?: string }[],
    maxSeqLen: number,
  ): Promise<RemoteKvHandle[]> {
    const handles: RemoteKvHandle[] = []
    const sessionId = randomUUID()
    try {
      for (const step of remoteSteps) {
        const pid = peerIdFromString(step.peerId)
        const client = await KVSessionClient.open(this.node.libp2p, pid, sessionId, maxSeqLen)
        handles.push({
          peerId: step.peerId,
          client,
          close: () => client.close(),
        })
      }
      return handles
    } catch (err) {
      for (const h of handles) {
        await h.close().catch(() => {})
      }
      throw err
    }
  }
}

// IPC wiring + broadcasts have moved to ./chat-session-ipc.ts so this module
// stays free of electron imports (allows bun to import it during tests).
