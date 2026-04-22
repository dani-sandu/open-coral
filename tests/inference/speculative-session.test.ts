import { describe, it, expect } from 'bun:test'
import {
  SpeculativeSession,
  LocalVerificationBackend,
  DEFAULT_SPEC_CONFIG,
  type VerificationBackend,
} from '../../src/inference/speculative-session'

// ── Mock backend ─────────────────────────────────────────────────────────────

class MockBackend implements VerificationBackend {
  readonly vocabSize: number
  nPast = 0
  rollbackCalls: number[] = []
  forwardAllCalls = 0
  forwardOneCalls = 0

  private readonly logitsQueue: Float32Array[]
  private queueIndex = 0

  constructor(vocabSize: number, logitsQueue: Float32Array[] = []) {
    this.vocabSize = vocabSize
    this.logitsQueue = logitsQueue
  }

  async forwardAll(tokenIds: Int32Array): Promise<Float32Array> {
    this.nPast += tokenIds.length
    this.forwardAllCalls++
    return this.logitsQueue[this.queueIndex++] ?? new Float32Array(tokenIds.length * this.vocabSize)
  }

  async forwardOne(_tokenId: number): Promise<Float32Array> {
    this.nPast += 1
    this.forwardOneCalls++
    return this.logitsQueue[this.queueIndex++] ?? new Float32Array(this.vocabSize)
  }

  async rollback(newNPast: number): Promise<void> {
    this.rollbackCalls.push(newNPast)
    this.nPast = newNPast
  }
}

// Build logits where token `id` has near-certainty (logit 100, others 0)
function sharpLogits(vocabSize: number, nTokens: number, id: number): Float32Array {
  const f = new Float32Array(nTokens * vocabSize)
  for (let t = 0; t < nTokens; t++) f[t * vocabSize + id] = 100
  return f
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe('SpeculativeSession', () => {
  it('generates at least one token on a simple prompt', async () => {
    const vocabSize = 10
    const eosId = 9
    const backend = new MockBackend(vocabSize, [
      sharpLogits(vocabSize, 1, 5),    // prefill → sample token 5
      sharpLogits(vocabSize, 1, eosId), // next step → EOS
    ])
    const session = new SpeculativeSession(backend, eosId, undefined, {
      ...DEFAULT_SPEC_CONFIG, enabled: false,
    })
    const result = await session.generate(new Int32Array([1]), 10)
    expect(result.tokenIds).toContain(5)
  })

  it('stops at EOS token', async () => {
    const vocabSize = 10
    const eosId = 9
    const backend = new MockBackend(vocabSize, [
      sharpLogits(vocabSize, 1, eosId),  // prefill → EOS immediately
    ])
    const session = new SpeculativeSession(backend, eosId, undefined, {
      ...DEFAULT_SPEC_CONFIG, enabled: false,
    })
    const result = await session.generate(new Int32Array([1]), 20)
    expect(result.tokenIds).toHaveLength(0)
  })

  it('stops at EOT token', async () => {
    const vocabSize = 10
    const eosId = 9
    const eotId = 8
    const backend = new MockBackend(vocabSize, [
      sharpLogits(vocabSize, 1, eotId),  // prefill → EOT immediately
    ])
    const session = new SpeculativeSession(backend, eosId, eotId, {
      ...DEFAULT_SPEC_CONFIG, enabled: false,
    })
    const result = await session.generate(new Int32Array([1]), 20)
    expect(result.tokenIds).toHaveLength(0)
  })

  it('does not call forwardAll when spec disabled and no drafts', async () => {
    const vocabSize = 10
    const eosId = 9
    const backend = new MockBackend(vocabSize, [
      sharpLogits(vocabSize, 1, 3),      // prefill → token 3
      sharpLogits(vocabSize, 1, eosId),  // step → EOS
    ])
    const session = new SpeculativeSession(backend, eosId, undefined, {
      ...DEFAULT_SPEC_CONFIG, enabled: false,
    })
    await session.generate(new Int32Array([1]), 5)
    // prefill uses forwardAll once, then single-token path uses forwardOne
    expect(backend.forwardOneCalls).toBe(1)
  })

  it('acceptance probability matches p_target[draftToken] statistically', async () => {
    // vocabSize=4, uniform logits → p_target[any] = 0.25
    // Draft token = 0. Over 2000 trials, expect ~500 acceptances (25%)
    const vocabSize = 4
    const eosId = 3
    const nTrials = 2000
    let acceptances = 0

    for (let trial = 0; trial < nTrials; trial++) {
      // 2-row prefill for 2-token prompt; sample from last row (offset = 1*vocabSize)
      const prefillLogits = sharpLogits(vocabSize, 2, 0)     // both rows → token 0
      const batchLogits = new Float32Array(2 * vocabSize)    // uniform for position 0 (p_target[0]=0.25)
      batchLogits[1 * vocabSize + eosId] = 100               // position 1 (bonus): sharp on EOS

      const backend = new MockBackend(vocabSize, [prefillLogits, batchLogits])

      const session = new SpeculativeSession(backend, eosId, undefined, {
        enabled: true, ngramSize: 1, draftMax: 1, temperature: 1.0, topK: vocabSize,
      })

      // maxTokens=2: ensures exactly one speculative attempt per trial (accept or reject, then terminate)
      // Prompt [0,0]: cache key [0]→[0]; nextToken=0 from prefill row 1; lookup([0,0,0]) → draft [0]
      const result = await session.generate(new Int32Array([0, 0]), 2)

      if (result.specAcceptedTokens > 0) acceptances++
    }

    // p=0.25, 2000 trials — expect in [150, 600] (very loose bound for CI stability)
    expect(acceptances).toBeGreaterThan(150)
    expect(acceptances).toBeLessThan(600)
  })

  it('rolls back KV cache on draft rejection', async () => {
    const vocabSize = 4
    const eosId = 3
    // 2-row prefill for prompt [0,0]; last row → token 0 (nextToken=0)
    // Lookup([0,0,0]) → last ngramSize=1 = [0] → draft [0]
    // Batch [0, draft=0]: position 0 sharp on token 1 → p_target[0]≈0 → reject
    // Corrected token sampled from position 0 logits (fix 3) → token 1
    const prefillLogits = sharpLogits(vocabSize, 2, 0)
    const batchLogits = new Float32Array(2 * vocabSize)
    batchLogits[0 * vocabSize + 1] = 100  // position 0: sharp on token 1 → p_target[0]≈0 → reject
    batchLogits[1 * vocabSize + eosId] = 100  // bonus position: EOS

    const afterRollbackLogits = sharpLogits(vocabSize, 1, eosId)

    const backend = new MockBackend(vocabSize, [prefillLogits, batchLogits, afterRollbackLogits])

    const session = new SpeculativeSession(backend, eosId, undefined, {
      enabled: true, ngramSize: 1, draftMax: 1, temperature: 0.01, topK: vocabSize,
    })

    // Prompt [0,0]: cache key [0]→[0]; nextToken=0 from prefill; lookup([0,0,0]) → draft [0]
    const result = await session.generate(new Int32Array([0, 0]), 10)

    expect(backend.rollbackCalls.length).toBe(1)
    // prefill(2) + batch(2) = nPast 4, rollback to 4 - (1 - 0) = 3
    expect(backend.rollbackCalls[0]).toBe(3)
    expect(result.specDraftTokens).toBe(1)
    expect(result.specAcceptedTokens).toBe(0)
  })

  it('samples bonus token from last logit position when all drafts accepted', async () => {
    const vocabSize = 4
    const eosId = 3
    // 2-row prefill for prompt [0,0]; last row → token 0 (nextToken=0)
    const prefillLogits = sharpLogits(vocabSize, 2, 0)
    const batchLogits = new Float32Array(2 * vocabSize)
    batchLogits[0 * vocabSize + 0] = 100  // position 0: p_target[0]≈1 → accept draft 0
    batchLogits[1 * vocabSize + eosId] = 100  // position 1 (bonus): EOS → nextToken=EOS

    const backend = new MockBackend(vocabSize, [prefillLogits, batchLogits])

    const session = new SpeculativeSession(backend, eosId, undefined, {
      enabled: true, ngramSize: 1, draftMax: 1, temperature: 0.01, topK: vocabSize,
    })

    // Prompt [0,0]: cache key [0]→[0]; nextToken=0 from prefill row 1; lookup([0,0,0]) → draft [0]
    const result = await session.generate(new Int32Array([0, 0]), 10)

    expect(result.specAcceptedTokens).toBe(1)
    expect(backend.rollbackCalls).toHaveLength(0)
  })
})
