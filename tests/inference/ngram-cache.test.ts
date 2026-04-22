import { describe, it, expect } from 'bun:test'
import { NgramCache } from '../../src/inference/ngram-cache'

describe('NgramCache', () => {
  it('returns empty array when no match found', () => {
    const cache = new NgramCache(4, 5)
    cache.buildFromTokens([1, 2, 3, 4, 5, 6])
    const drafts = cache.lookup([10, 11, 12, 13])
    expect(drafts).toEqual([])
  })

  it('finds a continuation when n-gram repeats', () => {
    const cache = new NgramCache(4, 5)
    cache.buildFromTokens([1, 2, 3, 4, 5, 6, 1, 2, 3, 4])
    const drafts = cache.lookup([1, 2, 3, 4])
    expect(drafts).toEqual([5, 6, 1, 2, 3])
  })

  it('caps continuation at draftMax', () => {
    const cache = new NgramCache(2, 3)
    cache.buildFromTokens([1, 2, 10, 11, 12, 13, 14])
    const drafts = cache.lookup([1, 2])
    expect(drafts.length).toBeLessThanOrEqual(3)
    expect(drafts).toEqual([10, 11, 12])
  })

  it('returns latest match when multiple n-gram occurrences exist', () => {
    const cache = new NgramCache(2, 3)
    cache.buildFromTokens([1, 2, 3, 4, 1, 2, 5, 6])
    const drafts = cache.lookup([1, 2])
    expect(drafts).toEqual([5, 6])
  })

  it('addToken incrementally extends the cache', () => {
    const cache = new NgramCache(2, 3)
    cache.buildFromTokens([1, 2, 3])
    cache.addToken(4, [1, 2, 3, 4])
    cache.addToken(5, [1, 2, 3, 4, 5])
    const drafts = cache.lookup([3, 4])
    expect(drafts).toEqual([5])
  })

  it('handles context shorter than ngramSize gracefully', () => {
    const cache = new NgramCache(4, 5)
    cache.buildFromTokens([1, 2])
    const drafts = cache.lookup([1, 2])
    expect(drafts).toEqual([])
  })

  it('lookup uses only the last ngramSize tokens from context', () => {
    const cache = new NgramCache(3, 5)
    cache.buildFromTokens([10, 20, 1, 2, 3, 4, 5])
    const drafts = cache.lookup([0, 0, 1, 2, 3])
    expect(drafts).toEqual([4, 5])
  })
})
