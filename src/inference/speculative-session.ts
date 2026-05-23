import { NgramCache } from './ngram-cache'
import { sampleTopK, softmaxProb } from './sampler'
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
}

export const DEFAULT_SPEC_CONFIG: SpecConfig = {
  enabled: true,
  ngramSize: 4,
  draftMax: 5,
  temperature: 0.7,
  topK: 40,
}

export interface GenerateResult {
  tokenIds: number[]
  specDraftTokens: number
  specAcceptedTokens: number
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
    const { temperature, topK, enabled, ngramSize, draftMax } = config

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

    while (budget.shouldContinue()) {
      if (nextToken === this.eosTokenId) break
      if (this.eotTokenId !== undefined && nextToken === this.eotTokenId) break

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
      const maxDrafts = Math.min(draftTokens.length, Math.max(0, budget.remaining() - 1))

      if (maxDrafts === 0) {
        // Single-token path
        generatedIds.push(nextToken)
        budget.count(nextToken)
        if (enabled) ngramCache.addToken(nextToken, allTokens)

        const logits = await backend.forwardOne(nextToken)
        nextToken = sampleTopK(logits, vocabSize, 0, temperature, topK)
      } else {
        // Speculative path: verify [nextToken, d0, d1, ...] in one batch
        const batch = new Int32Array(1 + maxDrafts)
        batch[0] = nextToken
        for (let i = 0; i < maxDrafts; i++) batch[i + 1] = draftTokens[i]

        specDraftTokens += maxDrafts
        const allLogits = await backend.forwardAll(batch)
        const kvPositionAfterBatch = backend.nPast

        // nextToken is always accepted (was already sampled from target distribution)
        generatedIds.push(nextToken)
        budget.count(nextToken)
        if (enabled) ngramCache.addToken(nextToken, allTokens)

        let accepted = 0
        let rejected = false
        let terminated = false  // EOS/EOT accepted as draft — stop outer loop without sampling bonus

        for (let i = 0; i < maxDrafts && budget.shouldContinue(); i++) {
          // Speculative sampling (Leviathan et al. 2023):
          // n-gram draft is a point mass on draftTokens[i], so accept with p_target[draftTokens[i]]
          const pAccept = softmaxProb(allLogits, i * vocabSize, vocabSize, draftTokens[i])

          if (Math.random() < pAccept) {
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
            // then rollback KV cache to discard the unaccepted draft slots.
            nextToken = sampleTopK(allLogits, vocabSize, i * vocabSize, temperature, topK)
            const rollbackTo = kvPositionAfterBatch - (maxDrafts - accepted)
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
            // KV is now in sync with the accepted prefix; trim allTokens to match.
            allTokens.length = rollbackTo
            rejected = true
            break
          }
        }

        if (terminated) break
        if (!rejected) {
          // All drafts accepted: sample bonus token from position after last draft
          nextToken = sampleTopK(allLogits, vocabSize, maxDrafts * vocabSize, temperature, topK)
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
    const { temperature, topK, enabled, ngramSize, draftMax } = config

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

    while (budget.shouldContinue()) {
      if (nextToken === this.eosTokenId) break
      if (this.eotTokenId !== undefined && nextToken === this.eotTokenId) break

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
      const maxDrafts = Math.min(draftTokens.length, Math.max(0, budget.remaining() - 1))

      if (maxDrafts === 0) {
        yield nextToken
        budget.count(nextToken)
        if (enabled) ngramCache.addToken(nextToken, allTokens)
        const logits = await backend.forwardOne(nextToken)
        nextToken = sampleTopK(logits, vocabSize, 0, temperature, topK)
      } else {
        const batch = new Int32Array(1 + maxDrafts)
        batch[0] = nextToken
        for (let i = 0; i < maxDrafts; i++) batch[i + 1] = draftTokens[i]

        const allLogits = await backend.forwardAll(batch)
        const kvPositionAfterBatch = backend.nPast

        yield nextToken
        budget.count(nextToken)
        if (enabled) ngramCache.addToken(nextToken, allTokens)

        let accepted = 0
        let rejected = false
        let terminated = false

        for (let i = 0; i < maxDrafts && budget.shouldContinue(); i++) {
          const pAccept = softmaxProb(allLogits, i * vocabSize, vocabSize, draftTokens[i])
          if (Math.random() < pAccept) {
            accepted++
            if (draftTokens[i] === this.eosTokenId) { terminated = true; break }
            if (this.eotTokenId !== undefined && draftTokens[i] === this.eotTokenId) { terminated = true; break }
            yield draftTokens[i]
            budget.count(draftTokens[i])
            allTokens.push(draftTokens[i])
            if (enabled) ngramCache.addToken(draftTokens[i], allTokens)
          } else {
            nextToken = sampleTopK(allLogits, vocabSize, i * vocabSize, temperature, topK)
            const rollbackTo = kvPositionAfterBatch - (maxDrafts - accepted)
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
            allTokens.length = rollbackTo
            rejected = true
            break
          }
        }

        if (terminated) break
        if (!rejected) {
          nextToken = sampleTopK(allLogits, vocabSize, maxDrafts * vocabSize, temperature, topK)
        }
      }
    }
  }
}
