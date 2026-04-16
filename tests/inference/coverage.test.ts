import { describe, it, expect } from 'bun:test'
import { computeCoverage, suggestBlockRange, type BlockCoverage } from '../../src/inference/coverage'
import type { BlockRange } from '../../src/inference/types'

function peer(blockStart: number, blockEnd: number, peerId = 'peerA'): BlockCoverage {
  return { blockStart, blockEnd, peerId, multiaddrs: [] }
}

describe('computeCoverage', () => {
  it('returns all blocks missing when no peers available', () => {
    const r = computeCoverage([], 3)
    expect(r.complete).toBe(false)
    expect(r.missing).toEqual([0, 1, 2])
    expect(r.covered).toHaveLength(0)
    expect(r.totalBlocks).toBe(3)
  })

  it('single range covering all blocks → complete', () => {
    const r = computeCoverage([peer(0, 2)], 3)
    expect(r.complete).toBe(true)
    expect(r.missing).toHaveLength(0)
    expect(r.covered).toHaveLength(1)
    expect(r.covered[0].peerId).toBe('peerA')
  })

  it('two adjacent non-overlapping ranges covering all blocks → complete', () => {
    const r = computeCoverage([peer(0, 1, 'A'), peer(2, 2, 'B')], 3)
    expect(r.complete).toBe(true)
    expect(r.missing).toHaveLength(0)
    expect(r.covered).toHaveLength(2)
    expect(r.covered.map(c => c.peerId)).toEqual(['A', 'B'])
  })

  it('gap in coverage → missing block reported', () => {
    const r = computeCoverage([peer(0, 0, 'A'), peer(2, 2, 'B')], 3)
    expect(r.complete).toBe(false)
    expect(r.missing).toEqual([1])
    expect(r.covered).toHaveLength(2)
  })

  it('prefers range with largest blockEnd at each position', () => {
    const r = computeCoverage([peer(0, 1, 'short'), peer(0, 2, 'long')], 3)
    expect(r.complete).toBe(true)
    expect(r.covered).toHaveLength(1)
    expect(r.covered[0].peerId).toBe('long')
  })

  it('totalBlocks 0 → empty complete report', () => {
    const r = computeCoverage([], 0)
    expect(r.complete).toBe(true)
    expect(r.missing).toHaveLength(0)
    expect(r.covered).toHaveLength(0)
  })

  it('range with blockEnd beyond totalBlocks is treated as full coverage without error', () => {
    const r = computeCoverage([peer(0, 99)], 3)
    expect(r.complete).toBe(true)
    expect(r.missing).toHaveLength(0)
    expect(r.covered).toHaveLength(1)
  })
})

describe('suggestBlockRange', () => {
  it('returns largest contiguous gap', () => {
    const report = computeCoverage([peer(0, 3, 'A'), peer(8, 9, 'B')], 10)
    const suggestion = suggestBlockRange(report)
    expect(suggestion).toEqual({ start: 4, end: 7 })
  })

  it('returns null when fully covered', () => {
    const report = computeCoverage([peer(0, 9)], 10)
    expect(suggestBlockRange(report)).toBeNull()
  })

  it('picks first gap when multiple gaps have equal size', () => {
    const report = computeCoverage([peer(0, 1, 'A'), peer(4, 5, 'B'), peer(8, 9, 'C')], 10)
    const suggestion = suggestBlockRange(report)
    expect(suggestion).toEqual({ start: 2, end: 3 })
  })

  it('handles all blocks missing', () => {
    const report = computeCoverage([], 4)
    const suggestion = suggestBlockRange(report)
    expect(suggestion).toEqual({ start: 0, end: 3 })
  })

  it('handles single missing block', () => {
    const report = computeCoverage([peer(0, 2, 'A'), peer(4, 4, 'B')], 5)
    const suggestion = suggestBlockRange(report)
    expect(suggestion).toEqual({ start: 3, end: 3 })
  })
})
