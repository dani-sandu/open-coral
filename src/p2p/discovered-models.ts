import type { PeerModelInfoPayload } from './model-announce'

function modelKey(repoId: string, hfFilename: string): string {
  return `${repoId}::${hfFilename}`
}

export interface NetworkModelEntry {
  repoId: string
  hfFilename: string
  totalBlocks: number
  hiddenSize: number
  architecture: string
  /** Number of unique block indices covered by at least one peer */
  coveredBlocks: number
  /** True if all blocks 0..(totalBlocks-1) are covered */
  complete: boolean
  peers: { peerId: string; blockStart: number; blockEnd: number }[]
}

export class DiscoveredModels {
  private readonly peers = new Map<string, PeerModelInfoPayload>()

  update(peerId: string, info: PeerModelInfoPayload): void {
    this.peers.set(peerId, info)
  }

  remove(peerId: string): void {
    this.peers.delete(peerId)
  }

  /** Return the block range for a specific peer, or null if not known. */
  getPeerRange(peerId: string): { blockStart: number; blockEnd: number } | null {
    const info = this.peers.get(peerId)
    if (!info) return null
    return { blockStart: info.blockStart, blockEnd: info.blockEnd }
  }

  list(): { peerId: string; info: PeerModelInfoPayload }[] {
    return [...this.peers.entries()].map(([peerId, info]) => ({ peerId, info }))
  }

  aggregate(): NetworkModelEntry[] {
    // Group peers by model identity
    const groups = new Map<string, {
      info: PeerModelInfoPayload
      peers: { peerId: string; blockStart: number; blockEnd: number }[]
    }>()

    for (const [peerId, info] of this.peers) {
      const key = modelKey(info.repoId, info.hfFilename)
      if (!groups.has(key)) {
        groups.set(key, { info, peers: [] })
      }
      groups.get(key)!.peers.push({ peerId, blockStart: info.blockStart, blockEnd: info.blockEnd })
    }

    const entries: NetworkModelEntry[] = []

    for (const [, { info, peers }] of groups) {
      // Compute union of covered block indices
      const covered = new Set<number>()
      for (const p of peers) {
        for (let i = p.blockStart; i <= p.blockEnd; i++) {
          covered.add(i)
        }
      }
      const coveredBlocks = covered.size
      entries.push({
        repoId:       info.repoId,
        hfFilename:   info.hfFilename,
        totalBlocks:  info.totalBlocks,
        hiddenSize:   info.hiddenSize,
        architecture: info.architecture,
        coveredBlocks,
        complete:     coveredBlocks >= info.totalBlocks,
        peers,
      })
    }

    // Sort: complete models first, then by coverage descending
    entries.sort((a, b) => {
      if (a.complete !== b.complete) return a.complete ? -1 : 1
      return b.coveredBlocks - a.coveredBlocks
    })

    return entries
  }
}
