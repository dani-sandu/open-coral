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
import { SpeculativeSession, LocalVerificationBackend, DEFAULT_SPEC_CONFIG, type GenerateResult } from '../inference/speculative-session'

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

      async onForwardAll(sessionId, input, nTokens, _nEmbd) {
        const entry = kvSessions.get(sessionId)
        if (!entry) throw new Error(`KV session not found: ${sessionId}`)
        entry.lastUsed = Date.now()
        return runner.projectToLogitsAll(input, nTokens)
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

    // Tokenize prompt with chat template
    const promptIds = await tokenizer.encodeChat(prompt)
    console.log(`[OpenCoral] [${requestId}] Prompt tokens (${promptIds.length})`)

    const shimSessionId = await embedRunner.openSession(promptIds.length + maxTokens)
    let inferenceDurationMs = 0
    let genResult: GenerateResult = { tokenIds: [], specDraftTokens: 0, specAcceptedTokens: 0 }

    try {
      const backend = new LocalVerificationBackend(embedRunner, shimSessionId)
      const session = new SpeculativeSession(
        backend,
        tokenizer.eosTokenId,
        tokenizer.endOfTurnTokenId,
        DEFAULT_SPEC_CONFIG,
      )
      const t1 = Date.now()
      genResult = await session.generate(new Int32Array(promptIds), maxTokens)
      inferenceDurationMs = Date.now() - t1

      if (genResult.specDraftTokens > 0) {
        const rate = (genResult.specAcceptedTokens / genResult.specDraftTokens * 100).toFixed(1)
        console.log(`[OpenCoral] [${requestId}] Spec decoding: ${genResult.specAcceptedTokens}/${genResult.specDraftTokens} accepted (${rate}%)`)
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

    return {
      prompt,
      generatedText: await tokenizer.decode(genResult.tokenIds),
      generatedTokens: genResult.tokenIds.length,
      nEmbd,
      chainSteps,
      totalDurationMs: Date.now() - t0,
      specDraftTokens: genResult.specDraftTokens || undefined,
      specAcceptedTokens: genResult.specAcceptedTokens || undefined,
      specAcceptanceRate: genResult.specDraftTokens > 0
        ? genResult.specAcceptedTokens / genResult.specDraftTokens
        : undefined,
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
