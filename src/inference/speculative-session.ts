import { NgramCache } from './ngram-cache'
import { sampleTopK, softmaxProb } from './sampler'
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

  constructor(
    backend: VerificationBackend,
    eosTokenId: number,
    eotTokenId: number | undefined,
    config: SpecConfig = DEFAULT_SPEC_CONFIG,
  ) {
    this.backend = backend
    this.eosTokenId = eosTokenId
    this.eotTokenId = eotTokenId
    this.config = config
  }

  async generate(promptIds: Int32Array, maxTokens: number): Promise<GenerateResult> {
    const { backend, config } = this
    const { vocabSize } = backend
    const { temperature, topK, enabled, ngramSize, draftMax } = config

    const generatedIds: number[] = []
    let specDraftTokens = 0
    let specAcceptedTokens = 0

    // Prefill: forwardAll returns logits for all prompt positions; sample from the last one.
    const prefillLogits = await backend.forwardAll(promptIds)
    let nextToken = sampleTopK(prefillLogits, vocabSize, (promptIds.length - 1) * vocabSize, temperature, topK)

    const allTokens: number[] = Array.from(promptIds)
    const ngramCache = new NgramCache(ngramSize, draftMax)
    if (enabled) ngramCache.buildFromTokens(allTokens)

    let totalGenerated = 0

    while (totalGenerated < maxTokens) {
      if (nextToken === this.eosTokenId) break
      if (this.eotTokenId !== undefined && nextToken === this.eotTokenId) break

      const draftTokens = enabled ? ngramCache.lookup([...allTokens, nextToken]) : []
      const maxDrafts = Math.min(draftTokens.length, maxTokens - totalGenerated - 1)

      if (maxDrafts === 0) {
        // Single-token path
        generatedIds.push(nextToken)
        allTokens.push(nextToken)
        if (enabled) ngramCache.addToken(nextToken, allTokens)
        totalGenerated++

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
        allTokens.push(nextToken)
        if (enabled) ngramCache.addToken(nextToken, allTokens)
        totalGenerated++

        let accepted = 0
        let rejected = false
        let terminated = false  // EOS/EOT accepted as draft — stop outer loop without sampling bonus

        for (let i = 0; i < maxDrafts && totalGenerated < maxTokens; i++) {
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
            allTokens.push(draftTokens[i])
            if (enabled) ngramCache.addToken(draftTokens[i], allTokens)
            totalGenerated++
          } else {
            // Draft rejected: sample corrected token from position i (the rejected slot's logits),
            // then rollback KV cache to discard the unaccepted draft slots.
            nextToken = sampleTopK(allLogits, vocabSize, i * vocabSize, temperature, topK)
            const rollbackTo = kvPositionAfterBatch - (maxDrafts - accepted)
            await backend.rollback(rollbackTo)
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
}
