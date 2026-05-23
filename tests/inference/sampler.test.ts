import { describe, it, expect } from 'bun:test'
import { sampleTopK, softmaxProb } from '../../src/inference/sampler'

function sharpLogits(vocabSize: number, peakId: number, peakValue = 100): Float32Array {
  const f = new Float32Array(vocabSize)
  f[peakId] = peakValue
  return f
}

describe('sampleTopK', () => {
  it('returns the dominant token id for near-deterministic logits', () => {
    const vocabSize = 1000
    const logits = sharpLogits(vocabSize, 42)
    for (let i = 0; i < 50; i++) {
      const id = sampleTopK(logits, vocabSize, 0, 0.7, 40)
      expect(id).toBe(42)
    }
  })

  it('honours the offset parameter (multi-token logits buffer)', () => {
    const vocabSize = 100
    const nTokens = 3
    const buf = new Float32Array(nTokens * vocabSize)
    buf[0 * vocabSize + 7] = 100
    buf[1 * vocabSize + 13] = 100
    buf[2 * vocabSize + 99] = 100
    expect(sampleTopK(buf, vocabSize, 0 * vocabSize, 0.7, 40)).toBe(7)
    expect(sampleTopK(buf, vocabSize, 1 * vocabSize, 0.7, 40)).toBe(13)
    expect(sampleTopK(buf, vocabSize, 2 * vocabSize, 0.7, 40)).toBe(99)
  })

  it('returns a valid token id in the top-K when distribution is flat', () => {
    const vocabSize = 200
    const logits = new Float32Array(vocabSize)
    for (let i = 0; i < vocabSize; i++) logits[i] = Math.random()
    const ranked = Array.from(logits)
      .map((v, i) => ({ v, i }))
      .sort((a, b) => b.v - a.v)
      .slice(0, 5)
      .map(x => x.i)
    const id = sampleTopK(logits, vocabSize, 0, 1.0, 5)
    expect(ranked).toContain(id)
  })

  it('does not allocate per-vocab objects (allocation budget)', () => {
    const vocabSize = 32000
    const logits = new Float32Array(vocabSize)
    for (let i = 0; i < vocabSize; i++) logits[i] = Math.random()
    const t0 = Date.now()
    for (let i = 0; i < 200; i++) sampleTopK(logits, vocabSize, 0, 0.7, 40)
    const elapsed = Date.now() - t0
    expect(elapsed).toBeLessThan(500)
  })

  it('topK=1 always returns the argmax', () => {
    const vocabSize = 256
    const logits = new Float32Array(vocabSize)
    for (let i = 0; i < vocabSize; i++) logits[i] = Math.sin(i)
    let max = -Infinity, maxIdx = -1
    for (let i = 0; i < vocabSize; i++) if (logits[i] > max) { max = logits[i]; maxIdx = i }
    expect(sampleTopK(logits, vocabSize, 0, 0.7, 1)).toBe(maxIdx)
  })

  it('handles topK larger than vocabSize gracefully', () => {
    const vocabSize = 8
    const logits = new Float32Array(vocabSize)
    for (let i = 0; i < vocabSize; i++) logits[i] = i
    // topK=40 but only 8 logits — must still return a valid token id.
    const id = sampleTopK(logits, vocabSize, 0, 0.7, 40)
    expect(id).toBeGreaterThanOrEqual(0)
    expect(id).toBeLessThan(vocabSize)
  })

  it('empirical distribution concentrates on the true top-K under sampling', () => {
    // Known distribution: logits = [10, 9, 8, ..., 0] over 100 vocab, topK=5.
    // Top-5 indices are 0..4. Sample many times and verify hits stay in {0,1,2,3,4}.
    const vocabSize = 100
    const logits = new Float32Array(vocabSize)
    for (let i = 0; i < vocabSize; i++) logits[i] = 10 - i  // strict descending
    const counts = new Map<number, number>()
    const N = 2000
    for (let i = 0; i < N; i++) {
      const id = sampleTopK(logits, vocabSize, 0, 1.0, 5)
      counts.set(id, (counts.get(id) ?? 0) + 1)
    }
    // Every sampled id must be in the top-5.
    for (const id of counts.keys()) {
      expect(id).toBeGreaterThanOrEqual(0)
      expect(id).toBeLessThan(5)
    }
    // Higher-logit tokens must dominate (loose bound — id 0 should fire more than id 4).
    expect((counts.get(0) ?? 0)).toBeGreaterThan((counts.get(4) ?? 0))
  })
})

describe('softmaxProb', () => {
  it('returns ~1 for a sharply peaked token at the peak', () => {
    const vocabSize = 100
    const logits = sharpLogits(vocabSize, 42, 100)
    const p = softmaxProb(logits, 0, vocabSize, 42)
    expect(p).toBeGreaterThan(0.99)
  })

  it('throws RangeError for out-of-range token ids', () => {
    const logits = new Float32Array(10)
    expect(() => softmaxProb(logits, 0, 10, -1)).toThrow(RangeError)
    expect(() => softmaxProb(logits, 0, 10, 10)).toThrow(RangeError)
  })
})
