import { describe, it, expect } from 'bun:test'
import { PeerLatencyTracker } from '../../src/p2p/peer-latency'

describe('PeerLatencyTracker', () => {
  it('returns Infinity for unknown peer', () => {
    const tracker = new PeerLatencyTracker()
    expect(tracker.getEstimate('unknown')).toBe(Infinity)
  })

  it('first sample sets the initial estimate', () => {
    const tracker = new PeerLatencyTracker()
    tracker.record('peer-A', 100)
    expect(tracker.getEstimate('peer-A')).toBe(100)
  })

  it('EWMA converges toward new samples', () => {
    const tracker = new PeerLatencyTracker()
    tracker.record('p', 100)
    tracker.record('p', 50)
    // EWMA α=0.2: estimate = 0.2×50 + 0.8×100 = 90
    expect(tracker.getEstimate('p')).toBeCloseTo(90, 5)
  })

  it('forget() removes peer estimates', () => {
    const tracker = new PeerLatencyTracker()
    tracker.record('p', 80)
    tracker.forget('p')
    expect(tracker.getEstimate('p')).toBe(Infinity)
  })

  it('bestPeer() returns the peer with lowest estimate', () => {
    const tracker = new PeerLatencyTracker()
    tracker.record('slow', 300)
    tracker.record('fast', 50)
    tracker.record('mid', 150)
    expect(tracker.bestPeer(['slow', 'fast', 'mid'])).toBe('fast')
  })

  it('bestPeer() returns first candidate when all are unknown', () => {
    const tracker = new PeerLatencyTracker()
    expect(tracker.bestPeer(['a', 'b', 'c'])).toBe('a')
  })

  it('bestPeer() returns null for empty candidate list', () => {
    const tracker = new PeerLatencyTracker()
    expect(tracker.bestPeer([])).toBeNull()
  })
})
