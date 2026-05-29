import { describe, it, expect } from 'bun:test'
import { SimNetwork } from '../src/sim/sim-network'
import { multiaddr } from '@multiformats/multiaddr'

describe('sim dial model (opt-in)', () => {
  it('default (dialLatencyMs=0): peers are pre-connected, dial is free', async () => {
    const net = new SimNetwork({ latencyMeanMs: 1, latencyJitterMs: 0, modelBlocks: 1 })
    const a = await net.addNode()
    const b = await net.addNode()
    expect(a.libp2p.getPeers().some((p: any) => p.equals(b.peerIdObj))).toBe(true)
  })

  it('dialLatencyMs>0: peers start disconnected; dial connects after a handshake delay', async () => {
    const net = new SimNetwork({ latencyMeanMs: 1, latencyJitterMs: 0, modelBlocks: 1, dialLatencyMs: 30 })
    const a = await net.addNode()
    const b = await net.addNode()
    expect(a.libp2p.getPeers().length).toBe(0)
    const t0 = Date.now()
    await a.libp2p.dial(multiaddr(b.multiaddrs[0]))
    const elapsed = Date.now() - t0
    expect(elapsed).toBeGreaterThanOrEqual(25)
    expect(a.libp2p.getPeers().some((p: any) => p.equals(b.peerIdObj))).toBe(true)
  })

  it('dial is idempotent: a second dial to a connected peer is instant', async () => {
    const net = new SimNetwork({ latencyMeanMs: 1, latencyJitterMs: 0, modelBlocks: 1, dialLatencyMs: 30 })
    const a = await net.addNode()
    const b = await net.addNode()
    await a.libp2p.dial(multiaddr(b.multiaddrs[0]))
    const t0 = Date.now()
    await a.libp2p.dial(multiaddr(b.multiaddrs[0]))
    expect(Date.now() - t0).toBeLessThan(10)
  })

  it('concurrent dials to the same peer share one handshake', async () => {
    const net = new SimNetwork({ latencyMeanMs: 1, latencyJitterMs: 0, modelBlocks: 1, dialLatencyMs: 30 })
    const a = await net.addNode()
    const b = await net.addNode()
    const t0 = Date.now()
    await Promise.all([
      a.libp2p.dial(multiaddr(b.multiaddrs[0])),
      a.libp2p.dial(multiaddr(b.multiaddrs[0])),
    ])
    expect(Date.now() - t0).toBeLessThan(60)
    expect(a.libp2p.getPeers().length).toBe(1)
  })
})
