const ALPHA = 0.2  // EWMA smoothing factor — higher = faster response to new samples

/**
 * Tracks per-peer round-trip time using exponential weighted moving average.
 * Unknown peers have an estimate of Infinity so they rank last during selection.
 */
export class PeerLatencyTracker {
  private readonly estimates = new Map<string, number>()

  /** Record a new RTT sample for a peer. Updates the EWMA estimate. */
  record(peerId: string, rttMs: number): void {
    const prev = this.estimates.get(peerId)
    if (prev === undefined) {
      this.estimates.set(peerId, rttMs)
    } else {
      this.estimates.set(peerId, ALPHA * rttMs + (1 - ALPHA) * prev)
    }
  }

  /** Return the current EWMA estimate for a peer, or Infinity if unknown. */
  getEstimate(peerId: string): number {
    return this.estimates.get(peerId) ?? Infinity
  }

  /** Remove all latency data for a peer (e.g. when they disconnect). */
  forget(peerId: string): void {
    this.estimates.delete(peerId)
  }

  /**
   * Return the peer ID with the lowest latency estimate from a list of candidates.
   * Returns the first candidate if all are unknown. Returns null for empty list.
   */
  bestPeer(candidates: string[]): string | null {
    if (candidates.length === 0) return null
    let best = candidates[0]
    let bestRtt = this.getEstimate(best)
    for (let i = 1; i < candidates.length; i++) {
      const rtt = this.getEstimate(candidates[i])
      if (rtt < bestRtt) {
        bestRtt = rtt
        best = candidates[i]
      }
    }
    return best
  }
}
