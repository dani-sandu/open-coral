import { describe, it, expect } from 'bun:test'
import { createTokenizer } from '../../src/inference/tokenizer'
import { GGUFValueType } from '../../src/inference/types'
import type { GGUFHeader } from '../../src/inference/types'

function makeHeader(tokens: string[], merges: string[], model = 'gpt2'): GGUFHeader {
  const metadata: GGUFHeader['metadata'] = [
    { key: 'tokenizer.ggml.model',       valueType: GGUFValueType.STRING, value: model },
    { key: 'tokenizer.ggml.tokens',       valueType: GGUFValueType.ARRAY,  value: tokens },
    { key: 'tokenizer.ggml.bos_token_id', valueType: GGUFValueType.UINT32, value: 0 },
    { key: 'tokenizer.ggml.eos_token_id', valueType: GGUFValueType.UINT32, value: 1 },
  ]
  if (merges.length > 0) {
    metadata.splice(2, 0, { key: 'tokenizer.ggml.merges', valueType: GGUFValueType.ARRAY, value: merges })
  }
  return {
    version: 3,
    tensorCount: 0n,
    metadata,
    tensors: [],
    metadataEndOffset: 100,
    dataRegionOffset: 1024n,
  }
}

const BOS = 0

describe('BPE tokenizer — merge-pair algorithm', () => {
  it('applies merge and produces bc instead of greedy ab+c for input "abc"', () => {
    // Vocab: bos(0) eos(1) c(2) a(3) ab(4) bc(5) b(6)
    // Merge: 'b c' → BPE of 'abc': [a,b,c] → merge b+c → [a,bc] → IDs [3,5]
    // Greedy would do: 'ab'(4) then 'c'(2) → [4,2]
    const vocab = ['<bos>', '<eos>', 'c', 'a', 'ab', 'bc', 'b']
    const tok = createTokenizer(makeHeader(vocab, ['b c']))
    const ids = Array.from(tok.encode('abc'))
    expect(ids[0]).toBe(BOS)
    expect(ids.slice(1)).toEqual([3, 5]) // a=3, bc=5
  })

  it('respects merge priority — lower index merge applied first', () => {
    // Vocab: bos(0) eos(1) a(2) b(3) c(4) ab(5) abc(6) bc(7)
    // Merges: ['a b', 'ab c'] → merge a+b first, then ab+c → [abc] → ID [6]
    const vocab = ['<bos>', '<eos>', 'a', 'b', 'c', 'ab', 'abc', 'bc']
    const tok = createTokenizer(makeHeader(vocab, ['a b', 'ab c']))
    const ids = Array.from(tok.encode('abc'))
    expect(ids[0]).toBe(BOS)
    expect(ids.slice(1)).toEqual([6]) // abc=6
  })

  it('falls back to greedy when no merge list is present', () => {
    const vocab = ['<bos>', '<eos>', 'a', 'b', 'ab']
    const tok = createTokenizer(makeHeader(vocab, []))  // no merges
    // Greedy: 'ab' found as longest match → [4]
    const ids = Array.from(tok.encode('ab'))
    expect(ids[0]).toBe(BOS)
    expect(ids.slice(1)).toEqual([4]) // ab=4
  })

  it('handles empty merge list (explicit but empty) as greedy fallback', () => {
    // Same as above but ensures empty array doesn't break anything
    const vocab = ['<bos>', '<eos>', 'a', 'b', 'ab']
    const header: GGUFHeader = {
      version: 3,
      tensorCount: 0n,
      metadata: [
        { key: 'tokenizer.ggml.model',       valueType: GGUFValueType.STRING, value: 'gpt2' },
        { key: 'tokenizer.ggml.tokens',       valueType: GGUFValueType.ARRAY,  value: vocab },
        { key: 'tokenizer.ggml.merges',       valueType: GGUFValueType.ARRAY,  value: [] },
        { key: 'tokenizer.ggml.bos_token_id', valueType: GGUFValueType.UINT32, value: 0 },
        { key: 'tokenizer.ggml.eos_token_id', valueType: GGUFValueType.UINT32, value: 1 },
      ],
      tensors: [],
      metadataEndOffset: 100,
      dataRegionOffset: 1024n,
    }
    const tok = createTokenizer(header)
    const ids = Array.from(tok.encode('ab'))
    expect(ids.slice(1)).toEqual([4])
  })
})
