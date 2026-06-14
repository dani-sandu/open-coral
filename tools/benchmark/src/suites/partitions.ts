import type { Partition } from '../types'

/** Convert sorted boundaries [0, b1, ..., B] into inclusive block ranges. */
export function boundariesToRanges(boundaries: number[]): { blockStart: number; blockEnd: number }[] {
  const ranges: { blockStart: number; blockEnd: number }[] = []
  for (let i = 0; i < boundaries.length - 1; i++) {
    ranges.push({ blockStart: boundaries[i], blockEnd: boundaries[i + 1] - 1 })
  }
  return ranges
}

/** Equal contiguous split of totalBlocks into nodeCount ranges (remainder spread to the front). */
function equalBoundaries(totalBlocks: number, nodeCount: number): number[] {
  const base = Math.floor(totalBlocks / nodeCount)
  const remainder = totalBlocks % nodeCount
  const boundaries = [0]
  let acc = 0
  for (let i = 0; i < nodeCount; i++) {
    acc += base + (i < remainder ? 1 : 0)
    boundaries.push(acc)
  }
  return boundaries
}

/**
 * Skew an equal split: the target node gets ~25% more blocks, taken evenly
 * from the others. Always returns strictly-increasing boundaries covering [0, totalBlocks].
 */
function skewBoundaries(totalBlocks: number, nodeCount: number, heavyIndex: number): number[] {
  if (nodeCount === 1) return [0, totalBlocks]
  const equal = equalBoundaries(totalBlocks, nodeCount)
  const sizes = boundariesToRanges(equal).map(r => r.blockEnd - r.blockStart + 1)
  const extra = Math.max(0, Math.round(sizes[heavyIndex] * 0.25))
  // Take `extra` blocks from the other nodes, round-robin, without emptying any.
  let taken = 0
  let i = 0
  while (taken < extra) {
    const idx = i % nodeCount
    i++
    if (idx === heavyIndex) continue
    if (sizes[idx] > 1) { sizes[idx]--; sizes[heavyIndex]++; taken++ }
    if (i > nodeCount * totalBlocks) break // safety
  }
  const boundaries = [0]
  let acc = 0
  for (const s of sizes) { acc += s; boundaries.push(acc) }
  return boundaries
}

export function generatePartitions(totalBlocks: number, nodeCount: number): Partition[] {
  return [
    { strategy: 'equal', boundaries: equalBoundaries(totalBlocks, nodeCount) },
    { strategy: 'front-heavy', boundaries: skewBoundaries(totalBlocks, nodeCount, 0) },
    { strategy: 'back-heavy', boundaries: skewBoundaries(totalBlocks, nodeCount, nodeCount - 1) },
  ]
}
