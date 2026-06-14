import { NgramCache } from './ngram-cache'
import { sampleTopK, marsAccept } from './sampler'
import { ThinkingBudget, THINKING_TOKEN_BUDGET, hasOpenThinkBlock, type ThinkTokens } from './thinking-budget'
import type { AsyncBlockRunner } from './native-worker'

export interface VerificationBackend {
  readonly vocabSize: number
  nPast: number
  forwardAll(tokenIds: Int32Array): Promise<Float32Array>
  forwardOne(tokenId: number): Promise<Float32Array>
  rollback(newNPast: number): Promise<void>
}

export interface SpecConfig {
  enabled: boolean
  ngramSize: number
  draftMax: number
  temperature: number
  topK: number
  /** MARS margin-aware accept: deterministically accept a draft within this ratio of the
   *  top token's probability (reduces rollbacks). 0 disables (pure speculative sampling). */
  marsMarginRatio?: number
  /** Adapt draft length to the recent acceptance rate (PEARL-style). When false, always
   *  attempts up to draftMax. Output distribution is unchanged either way. */
  adaptiveDraft?: boolean
  /** Minimum adaptive draft length (floor when acceptance is low). */
  draftMin?: number
  /** SpecPipe pipeline depth: 1 = serial (default, today's behavior); 2 = optimistic two-step
   *  pipelining (eagerly submit step N+1 while awaiting step N's verification). Only meaningful
   *  when the backend is a `PipelinedKVChain` — the LocalVerificationBackend ignores this. */
  pipelineDepth?: 1 | 2
}

export const DEFAULT_SPEC_CONFIG: SpecConfig = {
  enabled: true,
  ngramSize: 4,
  draftMax: 5,
  temperature: 0.7,
  topK: 40,
  marsMarginRatio: 0.9,
  adaptiveDraft: true,
  draftMin: 1,
  pipelineDepth: 1,
}

// EWMA smoothing for the running draft-acceptance signal that drives adaptive length.
const DRAFT_ACCEPT_ALPHA = 0.3

/**
 * Adaptive draft length (PEARL-style, adapted for the n-gram drafter which has no
 * per-token confidence): scale the draft cap between draftMin and draftMax by the
 * recent acceptance-rate EWMA. High acceptance → draft more; low → draft less to
 * avoid wasted forward+rollback. Output distribution is unchanged — this only sets
 * how many drafts are verified per step, not the accept/reject correctness.
 */
export function adaptiveDraftCap(acceptEwma: number, draftMin: number, draftMax: number): number {
  if (draftMax <= draftMin) return draftMax
  const clamped = Math.max(0, Math.min(1, acceptEwma))
  return Math.round(draftMin + clamped * (draftMax - draftMin))
}

export interface GenerateResult {
  tokenIds: number[]
  specDraftTokens: number
  specAcceptedTokens: number
}

function sameInt32(a: Int32Array, b: Int32Array): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

export class LocalVerificationBackend implements VerificationBackend {
  private readonly runner: AsyncBlockRunner
  private readonly sessionId: number
  nPast = 0

  constructor(runner: AsyncBlockRunner, sessionId: number) {
    this.runner = runner
    this.sessionId = sessionId
  }

  get vocabSize(): number {
    return this.runner.vocabSize
  }

  async forwardAll(tokenIds: Int32Array): Promise<Float32Array> {
    const logits = await this.runner.sessionDecodeLogitsAll(this.sessionId, tokenIds)
    this.nPast += tokenIds.length
    return logits
  }

  async forwardOne(tokenId: number): Promise<Float32Array> {
    const logits = await this.runner.sessionDecodeLogits(this.sessionId, new Int32Array([tokenId]))
    this.nPast += 1
    return logits
  }

  async rollback(newNPast: number): Promise<void> {
    await this.runner.sessionRollback(this.sessionId, newNPast)
    this.nPast = newNPast
  }
}

export class SpeculativeSession {
  private readonly backend: VerificationBackend
  private readonly eosTokenId: number
  private readonly eotTokenId: number | undefined
  private readonly config: SpecConfig
  private readonly thinkTokens: ThinkTokens | undefined
  private readonly thinkingBudget: number

  constructor(
    backend: VerificationBackend,
    eosTokenId: number,
    eotTokenId: number | undefined,
    config: SpecConfig = DEFAULT_SPEC_CONFIG,
    thinkTokens: ThinkTokens | undefined = undefined,
    thinkingBudget: number = THINKING_TOKEN_BUDGET,
  ) {
    this.backend = backend
    this.eosTokenId = eosTokenId
    this.eotTokenId = eotTokenId
    this.config = config
    this.thinkTokens = thinkTokens
    this.thinkingBudget = thinkingBudget
  }

  /**
   * @param contextTokens Tokens already present in the backend's KV cache before
   *   this call (e.g. conversation history pre-filled by ChatSessionManager).
   *   `allTokens` must mirror the full KV — the hybrid-model rollback fallback
   *   re-prefills from it — so callers that pre-fill MUST pass what they fed.
   *   Defaults to empty for callers that own the session from KV position 0.
   */
  async generate(
    promptIds: Int32Array,
    maxTokens: number,
    contextTokens: Int32Array = new Int32Array(0),
  ): Promise<GenerateResult> {
    const { backend, config } = this
    const { vocabSize } = backend
    const { temperature, topK, enabled, ngramSize, draftMax, marsMarginRatio = 0, adaptiveDraft = false, draftMin = 1, pipelineDepth = 1 } = config

    const generatedIds: number[] = []
    let specDraftTokens = 0
    let specAcceptedTokens = 0

    // Prefill: forwardAll returns logits for all prompt positions; sample from the last one.
    const prefillLogits = await backend.forwardAll(promptIds)
    let nextToken = sampleTopK(prefillLogits, vocabSize, (promptIds.length - 1) * vocabSize, temperature, topK)

    const allTokens: number[] = [...contextTokens, ...promptIds]
    const ngramCache = new NgramCache(ngramSize, draftMax)
    if (enabled) ngramCache.buildFromTokens(allTokens)

    // Hybrid/recurrent models (e.g. Qwen3.5) can't do partial KV rollback.
    // If the first rollback attempt fails we fall back to single-token generation.
    let speculationEnabled = enabled
    const SPEC_PREFILL_CHUNK = 256

    // Two-phase token budget: thinking tokens are capped separately from the
    // answer, so a reasoning model can never spend the answer cap on <think>.
    const budget = new ThinkingBudget(
      this.thinkTokens, this.thinkingBudget, maxTokens,
      hasOpenThinkBlock(allTokens, this.thinkTokens),
    )

    // Adaptive draft length (PEARL): running acceptance EWMA, starts optimistic.
    let acceptEwma = 1

    // Eager-submit (SpecPipe / P2-12): when pipelineDepth=2, pre-submit step N+1's
    // batch BEFORE awaiting step N — see below. Because pre-submit advances
    // backend.nPast optimistically, we track `safeNPast` as the logical
    // "guaranteed-verified" KV position independent of any in-flight pre-submit.
    let pendingForward: { batch: Int32Array; promise: Promise<Float32Array> } | null = null
    let safeNPast = backend.nPast

    while (budget.shouldContinue()) {
      if (nextToken === this.eosTokenId) break
      if (this.eotTokenId !== undefined && nextToken === this.eotTokenId) break

      // Logical iteration start — does NOT include any pending pre-submit writes.
      const kvPositionAtIterationStart = safeNPast

      // Thinking budget spent without a </think> — inject one to force the
      // model into the answer phase. The </think> is fed through the backend so
      // the KV cache reflects the phase change (reasoning models are trained to
      // switch to answering after </think>); it is emitted in the output but
      // not counted against either budget. The token sampled before this point
      // was a thinking continuation and is discarded.
      if (budget.needsForceClose() && this.thinkTokens !== undefined) {
        const closeId = this.thinkTokens.close
        generatedIds.push(closeId)
        allTokens.push(closeId)
        if (enabled) ngramCache.addToken(closeId, allTokens)
        const logits = await backend.forwardOne(closeId)
        budget.forceClose()
        nextToken = sampleTopK(logits, vocabSize, 0, temperature, topK)
        continue
      }

      // Push nextToken into the context FIRST so the lookup avoids a spread copy.
      // Invariants downstream (rollback math, ngramCache.addToken) line up because
      // nextToken was always destined to be pushed before any addToken call.
      allTokens.push(nextToken)
      const draftTokens = speculationEnabled ? ngramCache.lookup(allTokens) : []
      const draftCap = adaptiveDraft ? adaptiveDraftCap(acceptEwma, draftMin, draftMax) : draftMax
      const maxDrafts = Math.min(draftTokens.length, draftCap, Math.max(0, budget.remaining() - 1))

      if (maxDrafts === 0) {
        // Single-token path. Any in-flight pre-submit was speculating a
        // multi-token batch; its premise (matching speculative path) is wrong.
        if (pendingForward) {
          try { await pendingForward.promise } catch {}
          await backend.rollback(safeNPast)
          pendingForward = null
        }
        generatedIds.push(nextToken)
        budget.count(nextToken)
        if (enabled) ngramCache.addToken(nextToken, allTokens)

        const logits = await backend.forwardOne(nextToken)
        safeNPast += 1
        nextToken = sampleTopK(logits, vocabSize, 0, temperature, topK)
      } else {
        // Speculative path: verify [nextToken, d0, d1, ...] in one batch
        const batch = new Int32Array(1 + maxDrafts)
        batch[0] = nextToken
        for (let i = 0; i < maxDrafts; i++) batch[i + 1] = draftTokens[i]

        specDraftTokens += maxDrafts
        let nForwardPromise: Promise<Float32Array>
        if (pendingForward && pendingForward.batch.length === batch.length &&
            sameInt32(pendingForward.batch, batch)) {
          // The pending forward was speculatively submitted with this exact
          // batch — reuse its logits. backend.nPast was already advanced
          // optimistically at submit-time.
          nForwardPromise = pendingForward.promise
          pendingForward = null
        } else {
          if (pendingForward) {
            // We pre-submitted but the real batch differs (rare: budget shrank,
            // draftCap changed). Await + discard logits + rollback to a known
            // good position before submitting the real batch.
            try { await pendingForward.promise } catch {}
            await backend.rollback(kvPositionAtIterationStart)
            pendingForward = null
          }
          nForwardPromise = backend.forwardAll(batch)
        }

        // SpecPipe (P2-12) pre-submit BEFORE awaiting N: with depth=2, submit
        // N+1's batch concurrently so its hops overlap N's hops on the chain.
        // We don't yet know N's bonus token (logits haven't returned), so we
        // PREDICT it via ngram. If prediction matches actual on the next
        // iteration's batch check, we reuse the pre-submitted logits. Otherwise
        // the pendingForward is discarded + rolled back like any mismatched
        // pending. Output is unchanged either way.
        if (pipelineDepth === 2 && speculationEnabled && budget.shouldContinue()) {
          const speculativeAllTokens = allTokens.slice()
          for (let j = 0; j < maxDrafts; j++) speculativeAllTokens.push(draftTokens[j])
          const predictedNextCandidates = ngramCache.lookup(speculativeAllTokens)
          if (predictedNextCandidates.length > 0) {
            const predictedBonus = predictedNextCandidates[0]
            speculativeAllTokens.push(predictedBonus)
            const nextDrafts = ngramCache.lookup(speculativeAllTokens)
            const nextDraftCap = adaptiveDraft ? adaptiveDraftCap(acceptEwma, draftMin, draftMax) : draftMax
            const nextMaxDrafts = Math.min(nextDrafts.length, nextDraftCap, Math.max(0, budget.remaining() - 2 - maxDrafts))
            if (nextMaxDrafts > 0) {
              const nextBatch = new Int32Array(1 + nextMaxDrafts)
              nextBatch[0] = predictedBonus
              for (let j = 0; j < nextMaxDrafts; j++) nextBatch[j + 1] = nextDrafts[j]
              pendingForward = { batch: nextBatch, promise: backend.forwardAll(nextBatch) }
            }
          }
        }

        const allLogits = await nForwardPromise
        // Logical KV position after N's batch (independent of any in-flight
        // pre-submit for iter N+1 — backend.nPast may be ahead).
        const kvPositionAfterBatch = kvPositionAtIterationStart + batch.length

        // nextToken is always accepted (was already sampled from target distribution)
        generatedIds.push(nextToken)
        budget.count(nextToken)
        if (enabled) ngramCache.addToken(nextToken, allTokens)

        let accepted = 0
        let rejected = false
        let terminated = false  // EOS/EOT accepted as draft — stop outer loop without sampling bonus

        for (let i = 0; i < maxDrafts && budget.shouldContinue(); i++) {
          // Speculative sampling (Leviathan et al. 2023) + MARS margin-aware accept:
          // accept the n-gram draft stochastically with p_target[draft], OR deterministically
          // when it is a plausible runner-up within marsMarginRatio of the top probability.
          if (marsAccept(allLogits, i * vocabSize, vocabSize, draftTokens[i], marsMarginRatio)) {
            accepted++
            specAcceptedTokens++

            // Match non-speculative behavior: EOS/EOT terminates generation and is NOT emitted.
            if (draftTokens[i] === this.eosTokenId) { terminated = true; break }
            if (this.eotTokenId !== undefined && draftTokens[i] === this.eotTokenId) { terminated = true; break }

            generatedIds.push(draftTokens[i])
            budget.count(draftTokens[i])
            allTokens.push(draftTokens[i])
            if (enabled) ngramCache.addToken(draftTokens[i], allTokens)
          } else {
            // Draft rejected: sample corrected token from position i (the rejected slot's logits),
            // then rollback KV cache to discard the unaccepted draft slots AND any pre-submitted
            // iter-N+1 writes (whose all-accept premise was wrong).
            nextToken = sampleTopK(allLogits, vocabSize, i * vocabSize, temperature, topK)
            const rollbackTo = kvPositionAtIterationStart + 1 + accepted
            if (pendingForward) {
              try { await pendingForward.promise } catch {}
              pendingForward = null
            }
            try {
              await backend.rollback(rollbackTo)
            } catch {
              // Hybrid/recurrent models (e.g. Qwen3.5) can't partially roll back the KV
              // cache.  The native layer cleared it to 0; re-prefill from the accepted
              // prefix so the KV is consistent again, then disable speculation.
              backend.nPast = 0
              const accepted_tokens = new Int32Array(allTokens.slice(0, rollbackTo))
              for (let ri = 0; ri < rollbackTo; ri += SPEC_PREFILL_CHUNK) {
                await backend.forwardAll(accepted_tokens.subarray(ri, Math.min(ri + SPEC_PREFILL_CHUNK, rollbackTo)))
              }
              speculationEnabled = false
            }
            safeNPast = rollbackTo
            // KV is now in sync with the accepted prefix; trim allTokens to match.
            allTokens.length = rollbackTo
            rejected = true
            break
          }
        }

        if (adaptiveDraft && maxDrafts > 0) {
          acceptEwma = DRAFT_ACCEPT_ALPHA * (accepted / maxDrafts) + (1 - DRAFT_ACCEPT_ALPHA) * acceptEwma
        }

        if (terminated) break
        if (!rejected) {
          // All drafts accepted: sample bonus token from position after last draft
          nextToken = sampleTopK(allLogits, vocabSize, maxDrafts * vocabSize, temperature, topK)
          safeNPast = kvPositionAfterBatch
        }
      }
    }

    return { tokenIds: generatedIds, specDraftTokens, specAcceptedTokens }
  }

  /** Streaming counterpart of `generate`. See `generate` for `contextTokens`. */
  async *generateTokens(
    promptIds: Int32Array,
    maxTokens: number,
    contextTokens: Int32Array = new Int32Array(0),
  ): AsyncGenerator<number> {
    const { backend, config } = this
    const { vocabSize } = backend
    const { temperature, topK, enabled, ngramSize, draftMax, marsMarginRatio = 0, adaptiveDraft = false, draftMin = 1, pipelineDepth = 1 } = config

    const allTokens: number[] = [...contextTokens, ...promptIds]
    const ngramCache = new NgramCache(ngramSize, draftMax)
    if (enabled) ngramCache.buildFromTokens(allTokens)

    let speculationEnabled = enabled
    const SPEC_PREFILL_CHUNK = 256

    const prefillLogits = await backend.forwardAll(promptIds)
    let nextToken = sampleTopK(prefillLogits, vocabSize, (promptIds.length - 1) * vocabSize, temperature, topK)

    // Two-phase token budget — see `generate` for the rationale.
    const budget = new ThinkingBudget(
      this.thinkTokens, this.thinkingBudget, maxTokens,
      hasOpenThinkBlock(allTokens, this.thinkTokens),
    )

    // Adaptive draft length (PEARL): running acceptance EWMA, starts optimistic.
    let acceptEwma = 1

    let pendingForward: { batch: Int32Array; promise: Promise<Float32Array> } | null = null
    let safeNPast = backend.nPast

    while (budget.shouldContinue()) {
      if (nextToken === this.eosTokenId) break
      if (this.eotTokenId !== undefined && nextToken === this.eotTokenId) break

      // Logical iteration start — does NOT include any pending pre-submit writes.
      const kvPositionAtIterationStart = safeNPast

      // Thinking budget spent without a </think> — feed one through the backend
      // so the KV reflects the phase change, emit it, and switch to the answer
      // phase. The in-flight thinking token sampled before this is discarded.
      if (budget.needsForceClose() && this.thinkTokens !== undefined) {
        const closeId = this.thinkTokens.close
        yield closeId
        allTokens.push(closeId)
        if (enabled) ngramCache.addToken(closeId, allTokens)
        const logits = await backend.forwardOne(closeId)
        budget.forceClose()
        nextToken = sampleTopK(logits, vocabSize, 0, temperature, topK)
        continue
      }

      allTokens.push(nextToken)
      const draftTokens = speculationEnabled ? ngramCache.lookup(allTokens) : []
      const draftCap = adaptiveDraft ? adaptiveDraftCap(acceptEwma, draftMin, draftMax) : draftMax
      const maxDrafts = Math.min(draftTokens.length, draftCap, Math.max(0, budget.remaining() - 1))

      if (maxDrafts === 0) {
        if (pendingForward) {
          try { await pendingForward.promise } catch {}
          await backend.rollback(safeNPast)
          pendingForward = null
        }
        yield nextToken
        budget.count(nextToken)
        if (enabled) ngramCache.addToken(nextToken, allTokens)
        const logits = await backend.forwardOne(nextToken)
        safeNPast += 1
        nextToken = sampleTopK(logits, vocabSize, 0, temperature, topK)
      } else {
        const batch = new Int32Array(1 + maxDrafts)
        batch[0] = nextToken
        for (let i = 0; i < maxDrafts; i++) batch[i + 1] = draftTokens[i]

        let nForwardPromise: Promise<Float32Array>
        if (pendingForward && pendingForward.batch.length === batch.length &&
            sameInt32(pendingForward.batch, batch)) {
          nForwardPromise = pendingForward.promise
          pendingForward = null
        } else {
          if (pendingForward) {
            try { await pendingForward.promise } catch {}
            await backend.rollback(kvPositionAtIterationStart)
            pendingForward = null
          }
          nForwardPromise = backend.forwardAll(batch)
        }

        // SpecPipe (P2-12) pre-submit BEFORE awaiting N — see generate() for rationale.
        if (pipelineDepth === 2 && speculationEnabled && budget.shouldContinue()) {
          const speculativeAllTokens = allTokens.slice()
          for (let j = 0; j < maxDrafts; j++) speculativeAllTokens.push(draftTokens[j])
          const predictedNextCandidates = ngramCache.lookup(speculativeAllTokens)
          if (predictedNextCandidates.length > 0) {
            const predictedBonus = predictedNextCandidates[0]
            speculativeAllTokens.push(predictedBonus)
            const nextDrafts = ngramCache.lookup(speculativeAllTokens)
            const nextDraftCap = adaptiveDraft ? adaptiveDraftCap(acceptEwma, draftMin, draftMax) : draftMax
            const nextMaxDrafts = Math.min(nextDrafts.length, nextDraftCap, Math.max(0, budget.remaining() - 2 - maxDrafts))
            if (nextMaxDrafts > 0) {
              const nextBatch = new Int32Array(1 + nextMaxDrafts)
              nextBatch[0] = predictedBonus
              for (let j = 0; j < nextMaxDrafts; j++) nextBatch[j + 1] = nextDrafts[j]
              pendingForward = { batch: nextBatch, promise: backend.forwardAll(nextBatch) }
            }
          }
        }

        const allLogits = await nForwardPromise
        const kvPositionAfterBatch = kvPositionAtIterationStart + batch.length

        yield nextToken
        budget.count(nextToken)
        if (enabled) ngramCache.addToken(nextToken, allTokens)

        let accepted = 0
        let rejected = false
        let terminated = false

        for (let i = 0; i < maxDrafts && budget.shouldContinue(); i++) {
          if (marsAccept(allLogits, i * vocabSize, vocabSize, draftTokens[i], marsMarginRatio)) {
            accepted++
            if (draftTokens[i] === this.eosTokenId) { terminated = true; break }
            if (this.eotTokenId !== undefined && draftTokens[i] === this.eotTokenId) { terminated = true; break }
            yield draftTokens[i]
            budget.count(draftTokens[i])
            allTokens.push(draftTokens[i])
            if (enabled) ngramCache.addToken(draftTokens[i], allTokens)
          } else {
            nextToken = sampleTopK(allLogits, vocabSize, i * vocabSize, temperature, topK)
            const rollbackTo = kvPositionAtIterationStart + 1 + accepted
            if (pendingForward) {
              try { await pendingForward.promise } catch {}
              pendingForward = null
            }
            try {
              await backend.rollback(rollbackTo)
            } catch {
              backend.nPast = 0
              const acceptedTokens = new Int32Array(allTokens.slice(0, rollbackTo))
              for (let ri = 0; ri < rollbackTo; ri += SPEC_PREFILL_CHUNK) {
                await backend.forwardAll(acceptedTokens.subarray(ri, Math.min(ri + SPEC_PREFILL_CHUNK, rollbackTo)))
              }
              speculationEnabled = false
            }
            safeNPast = rollbackTo
            allTokens.length = rollbackTo
            rejected = true
            break
          }
        }

        if (adaptiveDraft && maxDrafts > 0) {
          acceptEwma = DRAFT_ACCEPT_ALPHA * (accepted / maxDrafts) + (1 - DRAFT_ACCEPT_ALPHA) * acceptEwma
        }

        if (terminated) break
        if (!rejected) {
          nextToken = sampleTopK(allLogits, vocabSize, maxDrafts * vocabSize, temperature, topK)
          safeNPast = kvPositionAfterBatch
        }
      }
    }
  }
}
