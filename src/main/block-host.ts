import { ipcMain } from 'electron'
import { randomUUID } from 'crypto'
import { AsyncBlockRunner } from '../inference/native-worker'
import { BlockRegistry } from '../p2p/block-registry'
import { registerInferenceHandler } from '../p2p/inference-protocol'
import { registerKVHandler, openRemoteSession, forwardRemote, closeRemoteSession, type KVSessionHandler } from '../p2p/kv-protocol'
import { getCurrentModel, getCurrentGGUFHeader } from './model-manager'
import { SequenceManager, type ChainStepWithCandidates } from '../inference/sequence-manager'
import type { CoverageReport } from '../inference/coverage'
import { createTokenizer, type Tokenizer } from '../inference/tokenizer'
import { peerIdFromString } from '@libp2p/peer-id'
import type { OpenCoralNode } from '../p2p/node'
import { getPeerBlockRange, getLatencyTracker } from './index'

export interface HostingState {
  modelPath: string
  blockStart: number
  blockEnd: number
  totalBlocks: number
  hiddenSize: number
}

// NOTE: Keep in sync with InferenceResult in src/renderer/src/types.ts.
// Duplication is intentional: Electron main and renderer processes cannot share
// TypeScript source directly; the IPC bridge serialises the value as plain JSON.
export interface InferenceResult {
  prompt: string
  generatedText: string
  generatedTokens: number
  nEmbd: number
  chainSteps: {
    peerId: string
    blockStart: number
    blockEnd: number
    durationMs: number
  }[]
  totalDurationMs: number
}

interface ActiveHost {
  runner: AsyncBlockRunner
  registry: BlockRegistry
  kvCleanupTimer: ReturnType<typeof setInterval>
  state: HostingState
}

let activeHost: ActiveHost | null = null
let cachedTokenizer: Tokenizer | null = null

/**
 * Sample a token from logits using temperature scaling and top-k filtering.
 * Returns the sampled token index.
 */
function sampleTopK(
  logits: Float32Array,
  vocabSize: number,
  offset: number,
  temperature: number = 0.7,
  topK: number = 40,
): number {
  // Build (index, logit) pairs for this token position
  const candidates: { idx: number; logit: number }[] = []
  for (let i = 0; i < vocabSize; i++) {
    candidates.push({ idx: i, logit: logits[offset + i] })
  }

  // Sort descending by logit
  candidates.sort((a, b) => b.logit - a.logit)

  // Keep only top-k
  const topCandidates = candidates.slice(0, topK)

  // Apply temperature and compute softmax
  const scaled = topCandidates.map(c => c.logit / temperature)
  const maxVal = scaled[0]
  const exps = scaled.map(v => Math.exp(v - maxVal))
  const sum = exps.reduce((a, b) => a + b, 0)
  const probs = exps.map(e => e / sum)

  // Sample from the distribution
  const r = Math.random()
  let cumulative = 0
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i]
    if (r <= cumulative) return topCandidates[i].idx
  }
  return topCandidates[topCandidates.length - 1].idx
}

// ── Block hosting IPC ─────────────────────────────────────────────────────────

export function setupBlockHostIPC(
  getNode: () => OpenCoralNode | null,
  onBlocksChanged: (blocks: { start: number; end: number }[]) => void,
): void {
  ipcMain.handle('opencoral:start-hosting', async (
    _event,
    blockStart: number,
    blockEnd: number,
  ): Promise<void> => {
    const model = getCurrentModel()
    if (!model) throw new Error('No model loaded — call opencoral:select-model first')

    const REQUIRED_SUFFIXES = [
      'attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight',
      'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight',
    ]
    const missing = REQUIRED_SUFFIXES.filter(s => !model.blockTensorSuffixes.includes(s))
    if (missing.length > 0) {
      const has = model.blockTensorSuffixes.join(', ')
      throw new Error(
        `This model uses an unsupported architecture (${model.architecture}). ` +
        `The native runner expects separate Q/K/V projections (llama-style) but this model has: ${has}. ` +
        `Missing: ${missing.join(', ')}`,
      )
    }

    const node = getNode()
    if (!node) throw new Error('P2P node not running')

    if (activeHost) {
      activeHost.runner.dispose()
      activeHost.registry.dispose()
      activeHost = null
    }
    cachedTokenizer = null  // invalidate on model/block change

    const runner = await AsyncBlockRunner.create({
      modelPath: model.path,
      blockStart,
      blockEnd,
      totalBlocks: model.totalBlocks,
      hiddenSize: model.hiddenSize,
    })

    const registry = new BlockRegistry(node.libp2p, {})
    await registry.start()

    await registerInferenceHandler(node.libp2p, async (input, nTokens) => {
      return runner.forward(input, nTokens)
    })

    // --- KV cache session handler ---
    const kvSessions = new Map<string, { sessionId: number; lastUsed: number }>()
    const KV_SESSION_TIMEOUT_MS = 5 * 60 * 1000  // 5 minutes

    const kvCleanupTimer = setInterval(() => {
      const now = Date.now()
      for (const [id, entry] of kvSessions) {
        if (now - entry.lastUsed > KV_SESSION_TIMEOUT_MS) {
          runner.closeSession(entry.sessionId).catch(() => {})
          kvSessions.delete(id)
          console.log(`[OpenCoral KV] Cleaned up stale session ${id}`)
        }
      }
    }, 60_000)

    const kvHandler: KVSessionHandler = {
      async onOpen(sessionId, maxSeqLen) {
        if (kvSessions.has(sessionId)) return { ok: true }  // idempotent
        const nativeSessionId = await runner.openSession(maxSeqLen)
        kvSessions.set(sessionId, { sessionId: nativeSessionId, lastUsed: Date.now() })
        return { ok: true }
      },

      async onForward(sessionId, input, nTokens, _nEmbd) {
        const entry = kvSessions.get(sessionId)
        if (!entry) throw new Error(`KV session not found: ${sessionId}`)
        entry.lastUsed = Date.now()
        return runner.sessionForward(entry.sessionId, input, nTokens)
      },

      async onClose(sessionId) {
        const entry = kvSessions.get(sessionId)
        if (!entry) return
        runner.closeSession(entry.sessionId).catch(() => {})
        kvSessions.delete(sessionId)
      },
    }

    await registerKVHandler(node.libp2p, kvHandler)

    activeHost = {
      runner,
      registry,
      kvCleanupTimer,
      state: {
        modelPath: model.path,
        blockStart,
        blockEnd,
        totalBlocks: model.totalBlocks,
        hiddenSize: model.hiddenSize,
      },
    }

    onBlocksChanged([{ start: blockStart, end: blockEnd }])
    console.log(`[OpenCoral] Now hosting blocks ${blockStart}–${blockEnd} from ${model.path}`)
  })

  ipcMain.handle('opencoral:stop-hosting', async (): Promise<void> => {
    if (!activeHost) return
    clearInterval(activeHost.kvCleanupTimer)
    await activeHost.runner.dispose()
    activeHost.registry.dispose()
    activeHost = null
    cachedTokenizer = null
    onBlocksChanged([])
    console.log('[OpenCoral] Stopped hosting blocks')
  })

  ipcMain.handle('opencoral:get-hosting-state', (): HostingState | null => {
    return activeHost?.state ?? null
  })
}

// ── Inference IPC ─────────────────────────────────────────────────────────────

export function setupInferenceIPC(getNode: () => OpenCoralNode | null): void {
  ipcMain.handle('opencoral:run-inference', async (
    _event,
    prompt: string,
    maxTokens: number,
  ): Promise<InferenceResult> => {
    if (typeof prompt !== 'string' || prompt.length === 0) {
      throw new Error('prompt must be a non-empty string')
    }
    if (!Number.isInteger(maxTokens) || maxTokens <= 0 || maxTokens > 256) {
      throw new Error(`maxTokens must be a positive integer <= 256, got ${maxTokens}`)
    }

    const model = getCurrentModel()
    if (!model) throw new Error('No model loaded')

    const node = getNode()
    if (!node) throw new Error('P2P node not running')

    if (!activeHost) throw new Error('No blocks hosted — start hosting first')

    // Build or reuse tokenizer from GGUF metadata
    if (!cachedTokenizer) {
      const header = getCurrentGGUFHeader()
      if (!header) throw new Error('GGUF header not available')
      cachedTokenizer = createTokenizer(header)
    }
    const tokenizer = cachedTokenizer

    const mgr = new SequenceManager({
      node,
      localRunner: activeHost.runner,
      totalBlocks: model.totalBlocks,
      hiddenSize: model.hiddenSize,
      getPeerBlockRange,
      latencyTracker: getLatencyTracker(),
    })

    const chain = await mgr.planChainWithCandidates()
    const nEmbd = model.hiddenSize

    const t0 = Date.now()
    const stepDurations = new Map<string, number>()
    const generatedIds: number[] = []

    // Tokenize prompt with chat template
    const promptIds = tokenizer.encodeChat(prompt)
    console.log(`[OpenCoral] Prompt tokens (${promptIds.length}):`, Array.from(promptIds).slice(0, 30))

    const vocabSize = activeHost.runner.vocabSize

    function stepKey(s: ChainStepWithCandidates): string {
      return `${s.candidates[0].peerId}:${s.blockStart}-${s.blockEnd}`
    }

    // Assign a KV session UUID for each remote chain step
    const remoteSessionIds = new Map<string, string>()
    const remoteSteps = chain.filter(s => s.candidates[0].peerId !== 'local')

    // Open KV sessions with all remote peers before prefill
    for (const step of remoteSteps) {
      const sessionUUID = randomUUID()
      remoteSessionIds.set(stepKey(step), sessionUUID)
      const peerId = peerIdFromString(step.candidates[0].peerId)
      await openRemoteSession(node.libp2p, peerId, sessionUUID, promptIds.length + maxTokens)
    }

    // Open a local KV cache session
    const sessionId = await activeHost.runner.openSession(promptIds.length + maxTokens)

    try {
      // ── Prefill: process full prompt ──────────────────────────────────────
      let current: Float32Array = await activeHost.runner.embedTokens(promptIds)

      for (const chainStep of chain) {
        const stepT0 = Date.now()
        if (chainStep.candidates[0].peerId === 'local') {
          current = await activeHost.runner.sessionForward(sessionId, current, promptIds.length)
        } else {
          const peerId = peerIdFromString(chainStep.candidates[0].peerId)
          const sessionUUID = remoteSessionIds.get(stepKey(chainStep))!
          current = await forwardRemote(node.libp2p, peerId, sessionUUID, current, promptIds.length, nEmbd)
        }
        stepDurations.set(stepKey(chainStep), (stepDurations.get(stepKey(chainStep)) ?? 0) + Date.now() - stepT0)
      }

      // Project only the last token's hidden state to logits
      const lastHidden = current.subarray((promptIds.length - 1) * nEmbd, promptIds.length * nEmbd)
      let logits = await activeHost.runner.projectToLogits(lastHidden, 1)
      let nextToken = sampleTopK(logits, vocabSize, 0, 0.7, 40)

      // ── Decode: one token at a time with KV cache ─────────────────────────
      for (let step = 0; step < maxTokens; step++) {
        if (nextToken === tokenizer.eosTokenId) break
        if (tokenizer.endOfTurnTokenId !== undefined && nextToken === tokenizer.endOfTurnTokenId) break

        generatedIds.push(nextToken)
        if (step < 5) {
          console.log(`[OpenCoral] Step ${step}: token=${nextToken} text="${tokenizer.decodeToken(nextToken)}"`)
        }

        // Embed single new token
        current = await activeHost.runner.embedTokens(new Int32Array([nextToken]))

        // Forward single token through chain (KV cached on both local and remote)
        for (const chainStep of chain) {
          const stepT0 = Date.now()
          if (chainStep.candidates[0].peerId === 'local') {
            current = await activeHost.runner.sessionForward(sessionId, current, 1)
          } else {
            const peerId = peerIdFromString(chainStep.candidates[0].peerId)
            const sessionUUID = remoteSessionIds.get(stepKey(chainStep))!
            current = await forwardRemote(node.libp2p, peerId, sessionUUID, current, 1, nEmbd)
          }
          stepDurations.set(stepKey(chainStep), (stepDurations.get(stepKey(chainStep)) ?? 0) + Date.now() - stepT0)
        }

        logits = await activeHost.runner.projectToLogits(current, 1)
        nextToken = sampleTopK(logits, vocabSize, 0, 0.7, 40)
      }
    } finally {
      await activeHost.runner.closeSession(sessionId)

      // Close remote KV sessions best-effort; host-side timeout handles dropout
      for (const step of remoteSteps) {
        const sessionUUID = remoteSessionIds.get(stepKey(step))
        if (!sessionUUID) continue
        const peerId = peerIdFromString(step.candidates[0].peerId)
        closeRemoteSession(node.libp2p, peerId, sessionUUID).catch(err => {
          console.warn(`[OpenCoral] Failed to close remote KV session for ${step.candidates[0].peerId}:`, err)
        })
      }
    }

    // Build chain step results from aggregated durations
    const chainSteps = chain.map(s => ({
      peerId: s.candidates[0].peerId,
      blockStart: s.blockStart,
      blockEnd: s.blockEnd,
      durationMs: stepDurations.get(stepKey(s)) ?? 0,
    }))

    return {
      prompt,
      generatedText: tokenizer.decode(generatedIds),
      generatedTokens: generatedIds.length,
      nEmbd,
      chainSteps,
      totalDurationMs: Date.now() - t0,
    }
  })
}

// ── Coverage IPC ──────────────────────────────────────────────────────────────

export function setupCoverageIPC(getNode: () => OpenCoralNode | null): void {
  ipcMain.handle('opencoral:check-coverage', async (): Promise<CoverageReport> => {
    const model = getCurrentModel()
    if (!model) {
      return { totalBlocks: 0, covered: [], missing: [], complete: false }
    }

    const node = getNode()
    if (!node) {
      return {
        totalBlocks: model.totalBlocks,
        covered: [],
        missing: Array.from({ length: model.totalBlocks }, (_, i) => i),
        complete: false,
      }
    }

    const mgr = new SequenceManager({
      node,
      localRunner: activeHost?.runner ?? null,
      totalBlocks: model.totalBlocks,
      hiddenSize: model.hiddenSize,
      getPeerBlockRange,
    })

    return mgr.checkCoverage()
  })
}

// ── Accessors ─────────────────────────────────────────────────────────────────

export function getActiveHost(): ActiveHost | null {
  return activeHost
}
