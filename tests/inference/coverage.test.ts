import { describe, it, expect } from 'bun:test'
import { computeCoverage, type BlockCoverage } from '../../src/inference/coverage'

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
