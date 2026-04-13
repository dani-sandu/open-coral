import { describe, it, expect, afterEach } from 'bun:test'
import { createOpenCoralNode, type OpenCoralNode } from '../../src/p2p/node'

describe('OpenCoralNode', () => {
  const nodes: OpenCoralNode[] = []
  afterEach(async () => {
    await Promise.all(nodes.map(n => n.stop()))
    nodes.length = 0
  })

  it('starts and exposes a peer ID', async () => {
    const node = await createOpenCoralNode()
    nodes.push(node)
    expect(node.peerId).toBeTypeOf('string')
    expect(node.peerId.length).toBeGreaterThan(10)
  })

  it('listens on at least one multiaddr', async () => {
    const node = await createOpenCoralNode()
    nodes.push(node)
    expect(node.multiaddrs.length).toBeGreaterThan(0)
    expect(node.multiaddrs[0]).toContain('/tcp/')
  })

  it('two nodes can dial each other', async () => {
    const a = await createOpenCoralNode()
    const b = await createOpenCoralNode()
    nodes.push(a, b)
    await a.libp2p.dial(b.libp2p.getMultiaddrs()[0])
    const peersOfA = a.libp2p.getPeers()
    expect(peersOfA.some(p => p.toString() === b.peerId)).toBe(true)
  })

  it('stop() resolves without error', async () => {
    const node = await createOpenCoralNode()
    await expect(node.stop()).resolves.toBeUndefined()
  })
})
