import { describe, it, expect } from 'bun:test'
import { SpeculativeSession, DEFAULT_SPEC_CONFIG, type VerificationBackend } from '../../src/inference/speculative-session'

class MockBackend implements VerificationBackend {
  readonly vocabSize: number
  nPast = 0
  private readonly logitsQueue: Float32Array[]
  private queueIndex = 0

  constructor(vocabSize: number, logitsQueue: Float32Array[] = []) {
    this.vocabSize = vocabSize
    this.logitsQueue = logitsQueue
  }

  async forwardAll(tokenIds: Int32Array): Promise<Float32Array> {
    this.nPast += tokenIds.length
    return this.logitsQueue[this.queueIndex++] ?? new Float32Array(tokenIds.length * this.vocabSize)
  }

  async forwardOne(_tokenId: number): Promise<Float32Array> {
    this.nPast += 1
    return this.logitsQueue[this.queueIndex++] ?? new Float32Array(this.vocabSize)
  }

  async rollback(newNPast: number): Promise<void> {
    this.nPast = newNPast
  }
}

function sharpLogits(vocabSize: number, nTokens: number, id: number): Float32Array {
  const f = new Float32Array(nTokens * vocabSize)
  for (let t = 0; t < nTokens; t++) f[t * vocabSize + id] = 100
  return f
}

describe('SpeculativeSession.generateTokens', () => {
  it('yields the same tokens as generate() produces', async () => {
    const vocabSize = 10
    const eosId = 9
    const logitsQueue = [
      sharpLogits(vocabSize, 1, 5),
      sharpLogits(vocabSize, 1, eosId),
    ]

    const backendA = new MockBackend(vocabSize, [...logitsQueue])
    const backendB = new MockBackend(vocabSize, [...logitsQueue])

    const sessionA = new SpeculativeSession(backendA, eosId, undefined, { ...DEFAULT_SPEC_CONFIG, enabled: false })
    const sessionB = new SpeculativeSession(backendB, eosId, undefined, { ...DEFAULT_SPEC_CONFIG, enabled: false })

    const result = await sessionA.generate(new Int32Array([1]), 10)
    const streamed: number[] = []
    for await (const id of sessionB.generateTokens(new Int32Array([1]), 10)) {
      streamed.push(id)
    }

    expect(streamed).toEqual(result.tokenIds)
  })

  it('yields nothing when first token is EOS', async () => {
    const vocabSize = 10
    const eosId = 9
    const backend = new MockBackend(vocabSize, [sharpLogits(vocabSize, 1, eosId)])
    const session = new SpeculativeSession(backend, eosId, undefined, { ...DEFAULT_SPEC_CONFIG, enabled: false })

    const tokens: number[] = []
    for await (const id of session.generateTokens(new Int32Array([1]), 20)) {
      tokens.push(id)
    }
    expect(tokens).toHaveLength(0)
  })

  it('stops at maxTokens', async () => {
    const vocabSize = 4
    const eosId = 3
    // all logits sharp on token 1 — no EOS until maxTokens hit
    const queue = Array.from({ length: 10 }, () => sharpLogits(vocabSize, 1, 1))
    const backend = new MockBackend(vocabSize, queue)
    const session = new SpeculativeSession(backend, eosId, undefined, { ...DEFAULT_SPEC_CONFIG, enabled: false })

    const tokens: number[] = []
    for await (const id of session.generateTokens(new Int32Array([0]), 3)) {
      tokens.push(id)
    }
    expect(tokens.length).toBeLessThanOrEqual(3)
  })
})
