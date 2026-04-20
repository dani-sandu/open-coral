import { describe, it, expect } from 'bun:test'
import { getNative } from '../../src/inference/native-loader'

const GGUF = process.env.TEST_GGUF_PATH

describe('LlamaBlockContext (native)', () => {
  it('skips all tests when TEST_GGUF_PATH is not set', () => {
    if (!GGUF) console.log('Skipping: set TEST_GGUF_PATH to run these tests')
  })

  it('loadBlockRange does not throw', () => {
    if (!GGUF) return
    const handle = getNative().loadBlockRange(GGUF, 0, 5, 28)
    expect(typeof handle).toBe('number')
    expect(handle).toBeGreaterThan(0)
    getNative().freeBlockRange(handle)
  })

  it('getVocabSize returns a positive number', () => {
    if (!GGUF) return
    const handle = getNative().loadBlockRange(GGUF, 0, 27, 28)
    const vocabSize = getNative().getVocabSize(handle)
    expect(vocabSize).toBeGreaterThan(0)
    getNative().freeBlockRange(handle)
  })

  it('embedTokens returns Float32Array with finite values', () => {
    if (!GGUF) return
    const handle = getNative().loadBlockRange(GGUF, 0, 27, 28)
    const ids = new Int32Array([1, 15339]) // BOS + a token
    const hidden = getNative().embedTokens(handle, ids)
    expect(hidden).toBeInstanceOf(Float32Array)
    expect(hidden.length).toBeGreaterThan(0)
    for (let i = 0; i < Math.min(hidden.length, 100); i++) {
      expect(isFinite(hidden[i])).toBe(true)
    }
    getNative().freeBlockRange(handle)
  })

  it('projectToLogits returns Float32Array of length vocabSize with finite values', () => {
    if (!GGUF) return
    const handle = getNative().loadBlockRange(GGUF, 0, 27, 28)
    const vocabSize = getNative().getVocabSize(handle)
    const ids = new Int32Array([1])
    const hidden = getNative().embedTokens(handle, ids)
    const logits = getNative().projectToLogits(handle, hidden, 1)
    expect(logits).toBeInstanceOf(Float32Array)
    expect(logits.length).toBe(vocabSize)
    for (let i = 0; i < Math.min(logits.length, 100); i++) {
      expect(isFinite(logits[i])).toBe(true)
    }
    getNative().freeBlockRange(handle)
  })
})
