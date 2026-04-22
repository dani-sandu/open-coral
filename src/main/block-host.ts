import { ipcMain } from 'electron'
import { randomUUID } from 'crypto'
import { AsyncBlockRunner } from '../inference/native-worker'
import { BlockRegistry } from '../p2p/block-registry'
import { registerInferenceHandlerV3 } from '../p2p/inference-protocol'
import { registerKVHandler, type KVSessionHandler } from '../p2p/kv-protocol'
import { getCurrentModel } from './model-manager'
import { SequenceManager } from '../inference/sequence-manager'
import { suggestBlockRange, type CoverageReport } from '../inference/coverage'
import { loadNativeTokenizer, freeNativeTokenizer, type NativeTokenizer } from '../inference/native-tokenizer'
import type { OpenCoralNode } from '../p2p/node'
import { getPeerBlockRange, getLatencyTracker, getNodeIdentity } from './index'
import { NgramCache } from '../inference/ngram-cache'

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
  specDraftTokens?: number
  specAcceptedTokens?: number
  specAcceptanceRate?: number
}

interface ActiveHost {
  runner: AsyncBlockRunner
  registry: BlockRegistry
  kvCleanupTimer: ReturnType<typeof setInterval>
  state: HostingState
}

let activeHost: ActiveHost | null = null
let cachedTokenizer: NativeTokenizer | null = null

// ── Shim runner for remote-only inference ────────────────────────────────────
// Loaded from a shim GGUF (embed + output tensors, no blocks).
// Used for embedTokens() and projectToLogits() when the user is not hosting blocks.
let shimRunner: AsyncBlockRunner | null = null

// ── Speculative decoding config ──────────────────────────────────────────────
const SPEC_ENABLED = true
const SPEC_NGRAM_SIZE = 12
const SPEC_DRAFT_MAX = 5

export async function loadShimRunner(shimPath: string, totalBlocks: number, hiddenSize: number): Promise<void> {
  if (shimRunner) {
    await shimRunner.dispose()
    shimRunner = null
  }
  if (cachedTokenizer) { await freeNativeTokenizer(cachedTokenizer); cachedTokenizer = null }
  cachedTokenizer = await loadNativeTokenizer(shimPath)
  shimRunner = await AsyncBlockRunner.create({
    modelPath: shimPath,
    blockStart: 0,
    blockEnd: -1,  // no blocks — only embed + output tensors
    totalBlocks,
    hiddenSize,
  })
  console.log(`[OpenCoral] Shim runner loaded from ${shimPath}`)
}

export async function disposeShimRunner(): Promise<void> {
  if (shimRunner) {
    await shimRunner.dispose()
    shimRunner = null
  }
  if (cachedTokenizer) { await freeNativeTokenizer(cachedTokenizer); cachedTokenizer = null }
}

export function getShimRunner(): AsyncBlockRunner | null {
  return shimRunner
}

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

    const node = getNode()
    if (!node) throw new Error('P2P node not running')

    if (activeHost) {
      activeHost.runner.dispose()
      activeHost.registry.dispose()
      activeHost = null
    }

    const runner = await AsyncBlockRunner.create({
      ...(model.shardFiles && model.shardFiles.length > 1
        ? { modelPaths: model.shardFiles }
        : { modelPath: model.path }
      ),
      blockStart,
      blockEnd,
      totalBlocks: model.totalBlocks,
      hiddenSize: model.hiddenSize,
    })

    const registry = new BlockRegistry(node.libp2p, model.repoId ?? 'unknown', {})
    await registry.start()

    await registerInferenceHandlerV3(node.libp2p, async (input, nTokens) => {
      return runner.forward(input, nTokens)
    }, getNodeIdentity())

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

      async onRollback(sessionId, newNPast) {
        const entry = kvSessions.get(sessionId)
        if (!entry) throw new Error(`KV session not found: ${sessionId}`)
        entry.lastUsed = Date.now()
        await runner.sessionRollback(entry.sessionId, newNPast)
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
    if (!Number.isInteger(maxTokens) || maxTokens <= 0 || maxTokens > 2048) {
      throw new Error(`maxTokens must be a positive integer <= 2048, got ${maxTokens}`)
    }

    const model = getCurrentModel()
    if (!model) throw new Error('No model loaded')

    const node = getNode()
    if (!node) throw new Error('P2P node not running')

    if (!activeHost && !shimRunner) throw new Error('No model ready — select a model first')

    const requestId = randomUUID()
    console.log(`[OpenCoral] Starting inference ${requestId}: "${prompt.slice(0, 50)}..."`)

    if (!cachedTokenizer) throw new Error('Tokenizer not loaded — call loadShimRunner first')
    const tokenizer = cachedTokenizer

    // Prefer activeHost runner for block computation; fall back to null for shim-only mode
    const localRunner = activeHost?.runner ?? null
    // For embed/project: prefer shimRunner (guaranteed to have embed + output tensors).
    // activeHost.runner may not have them if hosting a partial block range.
    const embedRunner = shimRunner ?? activeHost?.runner
    if (!embedRunner) throw new Error('No runner available for embed/project')

    const mgr = new SequenceManager({
      node,
      localRunner,
      totalBlocks: model.totalBlocks,
      hiddenSize: model.hiddenSize,
      getPeerBlockRange,
      latencyTracker: getLatencyTracker(),
      identity: getNodeIdentity(),
      repoId: model.repoId,
    })

    const chain = await mgr.planChainWithCandidates()
    const nEmbd = model.hiddenSize

    const t0 = Date.now()
    const generatedIds: number[] = []

    // Tokenize prompt with chat template
    const promptIds = await tokenizer.encodeChat(prompt)
    console.log(`[OpenCoral] [${requestId}] Prompt tokens (${promptIds.length})`)

    const vocabSize = embedRunner.vocabSize

    // shimRunner always loads the full model (block_range is informational-only in the
    // current patch — it does not filter tensors from a single GGUF file).
    // sessionDecodeLogits runs all transformer blocks with proper KV caching.
    // activeHost.runner serves REMOTE peer requests via the KV protocol; it is not
    // used in the local inference loop until true per-block shard files exist.
    const shimSessionId = await embedRunner.openSession(promptIds.length + maxTokens)
    let inferenceDurationMs = 0

    let specDraftTokens = 0
    let specAcceptedTokens = 0

    try {
      // ── Prefill ───────────────────────────────────────────────────────────
      const prefillT0 = Date.now()
      let logits = await embedRunner.sessionDecodeLogits(shimSessionId, promptIds)
      inferenceDurationMs += Date.now() - prefillT0
      let nextToken = sampleTopK(logits, vocabSize, 0, 0.7, 40)

      // Build n-gram cache from prompt tokens
      const allTokens: number[] = Array.from(promptIds)
      const ngramCache = new NgramCache(SPEC_NGRAM_SIZE, SPEC_DRAFT_MAX)
      if (SPEC_ENABLED) {
        ngramCache.buildFromTokens(allTokens)
      }

      // Track KV cache position for rollback (prefill consumed promptIds.length tokens)
      let kvPosition = promptIds.length

      // ── Decode with speculative drafting ──────────────────────────────────
      let totalGenerated = 0
      while (totalGenerated < maxTokens) {
        if (nextToken === tokenizer.eosTokenId) break
        if (tokenizer.endOfTurnTokenId !== undefined && nextToken === tokenizer.endOfTurnTokenId) break

        // Try to draft tokens from n-gram cache
        const context = [...allTokens, nextToken]
        const draftTokens = SPEC_ENABLED ? ngramCache.lookup(context) : []

        if (draftTokens.length === 0) {
          // ── No draft: single-token path (unchanged behavior) ────────────
          generatedIds.push(nextToken)
          allTokens.push(nextToken)
          if (SPEC_ENABLED) ngramCache.addToken(nextToken, allTokens)
          totalGenerated++

          if (totalGenerated <= 5) {
            console.log(`[OpenCoral] [${requestId}] Step ${totalGenerated}: token=${nextToken} text="${await tokenizer.decodeToken(nextToken)}"`)
          }

          const stepT0 = Date.now()
          logits = await embedRunner.sessionDecodeLogits(shimSessionId, new Int32Array([nextToken]))
          inferenceDurationMs += Date.now() - stepT0
          kvPosition++
          nextToken = sampleTopK(logits, vocabSize, 0, 0.7, 40)
        } else {
          // ── Speculative path: draft + verify ────────────────────────────
          // Cap drafts to remaining budget
          const maxDrafts = Math.min(draftTokens.length, maxTokens - totalGenerated - 1)
          const batch = new Int32Array(1 + maxDrafts)
          batch[0] = nextToken
          for (let i = 0; i < maxDrafts; i++) batch[i + 1] = draftTokens[i]

          specDraftTokens += maxDrafts

          const stepT0 = Date.now()
          const allLogits = await embedRunner.sessionDecodeLogitsAll(shimSessionId, batch)
          inferenceDurationMs += Date.now() - stepT0
          kvPosition += batch.length

          // Verify each draft token
          let accepted = 0
          generatedIds.push(nextToken)
          allTokens.push(nextToken)
          if (SPEC_ENABLED) ngramCache.addToken(nextToken, allTokens)
          totalGenerated++

          for (let i = 0; i < maxDrafts && totalGenerated < maxTokens; i++) {
            const targetToken = sampleTopK(allLogits, vocabSize, i * vocabSize, 0.7, 40)
            if (targetToken === draftTokens[i]) {
              // Draft accepted
              accepted++
              specAcceptedTokens++
              generatedIds.push(draftTokens[i])
              allTokens.push(draftTokens[i])
              if (SPEC_ENABLED) ngramCache.addToken(draftTokens[i], allTokens)
              totalGenerated++

              if (draftTokens[i] === tokenizer.eosTokenId) break
              if (tokenizer.endOfTurnTokenId !== undefined && draftTokens[i] === tokenizer.endOfTurnTokenId) break
            } else {
              // Draft rejected — sample correct token from this position
              // and rollback KV cache past accepted tokens
              const rollbackTo = kvPosition - (maxDrafts - accepted)
              await embedRunner.sessionRollback(shimSessionId, rollbackTo)
              kvPosition = rollbackTo

              nextToken = targetToken
              break
            }
          }

          if (accepted === maxDrafts) {
            // All drafts accepted — sample next token from last position's logits
            nextToken = sampleTopK(allLogits, vocabSize, maxDrafts * vocabSize, 0.7, 40)
          }
        }
      }
    } finally {
      await embedRunner.closeSession(shimSessionId)
    }

    // Build chain step results (block ranges from plan, total time from inference)
    const chainSteps = chain.map((s, i) => ({
      peerId: s.candidates[0].peerId,
      blockStart: s.blockStart,
      blockEnd: s.blockEnd,
      durationMs: i === 0 ? inferenceDurationMs : 0,
    }))

    const specAcceptanceRate = specDraftTokens > 0 ? specAcceptedTokens / specDraftTokens : 0
    if (specDraftTokens > 0) {
      console.log(`[OpenCoral] [${requestId}] Speculative decoding: ${specAcceptedTokens}/${specDraftTokens} accepted (${(specAcceptanceRate * 100).toFixed(1)}%)`)
    }

    return {
      prompt,
      generatedText: await tokenizer.decode(generatedIds),
      generatedTokens: generatedIds.length,
      nEmbd,
      chainSteps,
      totalDurationMs: Date.now() - t0,
      specDraftTokens: specDraftTokens || undefined,
      specAcceptedTokens: specAcceptedTokens || undefined,
      specAcceptanceRate: specDraftTokens > 0 ? specAcceptanceRate : undefined,
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
      identity: getNodeIdentity(),
      repoId: model.repoId,
    })

    const report = await mgr.checkCoverage()
    const suggestion = suggestBlockRange(report)
    return { ...report, suggestion: suggestion ?? undefined }
  })
}

// ── Accessors ─────────────────────────────────────────────────────────────────

export function getActiveHost(): ActiveHost | null {
  return activeHost
}
