import { describe, it, expect } from 'bun:test'
import type { ChatTurn } from '../../src/inference/native-tokenizer'

const GGUF_PATH = process.env.TEST_GGUF_PATH

describe('NativeTokenizer', async () => {
  if (!GGUF_PATH) {
    it.skip('skipped — set TEST_GGUF_PATH to run native tokenizer tests', () => {})
    return
  }

  // Dynamic import so the native module is only loaded when the env var is set
  const { loadNativeTokenizer, freeNativeTokenizer } = await import('../../src/inference/native-tokenizer')

  const tok = loadNativeTokenizer(GGUF_PATH)

  it('loads without throwing', () => {
    expect(tok).toBeDefined()
  })

  it('has a positive vocab size', () => {
    expect(tok.vocabSize).toBeGreaterThan(0)
  })

  it('has valid BOS and EOS token IDs', () => {
    expect(tok.bosTokenId).toBeGreaterThanOrEqual(0)
    expect(tok.bosTokenId).toBeLessThan(tok.vocabSize)
    expect(tok.eosTokenId).toBeGreaterThanOrEqual(0)
    expect(tok.eosTokenId).toBeLessThan(tok.vocabSize)
  })

  it('encode returns an Int32Array starting with BOS', () => {
    const ids = tok.encode('Hello world')
    expect(ids).toBeInstanceOf(Int32Array)
    expect(ids.length).toBeGreaterThan(1)
    expect(ids[0]).toBe(tok.bosTokenId)
  })

  it('decodeToken returns a string for BOS (may be empty)', () => {
    const piece = tok.decodeToken(tok.bosTokenId)
    expect(typeof piece).toBe('string')
  })

  it('decode round-trips a simple ASCII string approximately', () => {
    const text = 'Hello world'
    const ids = Array.from(tok.encode(text)).slice(1) // skip BOS
    const decoded = tok.decode(ids)
    expect(decoded.trim()).toContain('Hello')
  })

  it('encodeChat returns an Int32Array with more tokens than plain encode', () => {
    const plain = tok.encode('Hello')
    const chat  = tok.encodeChat('Hello')
    expect(chat).toBeInstanceOf(Int32Array)
    // Chat template wraps the message so it should produce at least as many tokens
    expect(chat.length).toBeGreaterThanOrEqual(plain.length)
  })

  it('freeNativeTokenizer does not throw', () => {
    const tok2 = loadNativeTokenizer(GGUF_PATH)
    expect(() => freeNativeTokenizer(tok2)).not.toThrow()
  })
})

describe('NativeTokenizer.encodeChatMulti', async () => {
  if (!GGUF_PATH) {
    it.skip('skipped — set TEST_GGUF_PATH to run native tokenizer tests', () => {})
    return
  }

  const { loadNativeTokenizer } = await import('../../src/inference/native-tokenizer')

  const tok = await loadNativeTokenizer(GGUF_PATH)

  it('produces a longer token sequence for multi-turn than single-turn', async () => {
    const single = await tok.encodeChat('Hello')
    const multi: ChatTurn[] = [
      { role: 'user',      content: 'Hello' },
      { role: 'assistant', content: 'Hi! How can I help?' },
      { role: 'user',      content: 'What is 2+2?' },
    ]
    const multiTokens = await tok.encodeChatMulti(multi)
    expect(multiTokens.length).toBeGreaterThan(single.length)
  })

  it('returns identical tokens when given a single user turn', async () => {
    const single = await tok.encodeChat('Hi there')
    const multi  = await tok.encodeChatMulti([{ role: 'user', content: 'Hi there' }])
    expect(Array.from(multi)).toEqual(Array.from(single))
  })
})
