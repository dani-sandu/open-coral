export interface BlockCoverage {
  blockStart: number
  blockEnd: number
  /** 'local' for the local BlockRunner, or a remote peer ID string */
  peerId: string
  multiaddrs: string[]
}

export interface CoverageReport {
  totalBlocks: number
  covered: BlockCoverage[]
  /** Block indices (0-based) with no coverage from any peer */
  missing: number[]
  complete: boolean
}

/**
 * Pure function: given a list of announced block ranges and the total block
 * count, compute which blocks are covered and which are missing.
 *
 * Uses a greedy algorithm: at each uncovered position, picks the available
 * range with the largest blockEnd that covers that position, then advances
 * past it.
 */
export function computeCoverage(
  available: BlockCoverage[],
  totalBlocks: number,
): CoverageReport {
  const covered: BlockCoverage[] = []
  const missing: number[] = []
  let position = 0

  while (position < totalBlocks) {
    const candidates = available.filter(
      r => r.blockStart <= position && r.blockEnd >= position,
    )

    if (candidates.length === 0) {
      missing.push(position)
      position++
    } else {
      const best = candidates.reduce((a, b) => (b.blockEnd > a.blockEnd ? b : a))
      covered.push(best)
      position = Math.min(best.blockEnd, totalBlocks - 1) + 1
    }
  }

  return { totalBlocks, covered, missing, complete: missing.length === 0 }
}
