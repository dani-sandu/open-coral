import { describe, it, expect, afterEach } from 'bun:test'
import { createCoralNode, type CoralNode } from '../../src/p2p/node'

describe('CoralNode', () => {
  const nodes: CoralNode[] = []
  afterEach(async () => {
    await Promise.all(nodes.map(n => n.stop()))
    nodes.length = 0
  })

  it('starts and exposes a peer ID', async () => {
    const node = await createCoralNode()
    nodes.push(node)
    expect(node.peerId).toBeTypeOf('string')
    expect(node.peerId.length).toBeGreaterThan(10)
  })

  it('listens on at least one multiaddr', async () => {
    const node = await createCoralNode()
    nodes.push(node)
    expect(node.multiaddrs.length).toBeGreaterThan(0)
    expect(node.multiaddrs[0]).toContain('/tcp/')
  })

  it('two nodes can dial each other', async () => {
    const a = await createCoralNode()
    const b = await createCoralNode()
    nodes.push(a, b)
    await a.libp2p.dial(b.libp2p.getMultiaddrs()[0])
    const peersOfA = a.libp2p.getPeers()
    expect(peersOfA.some(p => p.toString() === b.peerId)).toBe(true)
  })

  it('stop() resolves without error', async () => {
    const node = await createCoralNode()
    await expect(node.stop()).resolves.toBeUndefined()
  })
})
