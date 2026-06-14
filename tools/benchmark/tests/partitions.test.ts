import { describe, it, expect } from 'bun:test'
import { generatePartitions, boundariesToRanges } from '../src/suites/partitions'

function assertCovers(boundaries: number[], totalBlocks: number) {
  expect(boundaries[0]).toBe(0)
  expect(boundaries[boundaries.length - 1]).toBe(totalBlocks)
  for (let i = 1; i < boundaries.length; i++) {
    expect(boundaries[i]).toBeGreaterThan(boundaries[i - 1]) // strictly increasing → no gaps/overlaps/empty ranges
  }
}

describe('generatePartitions', () => {
  it('produces equal / front-heavy / back-heavy partitions that fully cover the model', () => {
    const totalBlocks = 32
    const partitions = generatePartitions(totalBlocks, 4)
    const strategies = partitions.map(p => p.strategy).sort()
    expect(strategies).toEqual(['back-heavy', 'equal', 'front-heavy'])
    for (const p of partitions) assertCovers(p.boundaries, totalBlocks)
  })

  it('equal split of 32 blocks over 4 nodes gives 4 ranges of 8', () => {
    const equal = generatePartitions(32, 4).find(p => p.strategy === 'equal')!
    expect(equal.boundaries).toEqual([0, 8, 16, 24, 32])
  })

  it('boundariesToRanges yields inclusive [start,end] ranges', () => {
    expect(boundariesToRanges([0, 8, 24, 32])).toEqual([
      { blockStart: 0, blockEnd: 7 },
      { blockStart: 8, blockEnd: 23 },
      { blockStart: 24, blockEnd: 31 },
    ])
  })

  it('handles nodeCount of 1 (single range covers everything)', () => {
    const partitions = generatePartitions(10, 1)
    for (const p of partitions) assertCovers(p.boundaries, 10)
    expect(generatePartitions(10, 1).find(p => p.strategy === 'equal')!.boundaries).toEqual([0, 10])
  })
})
