import { describe, it, expect, beforeAll, afterAll } from 'bun:test'
import { createCoralNode, type CoralNode } from '../../src/p2p/node'
import { announceBlocks, findPeerForBlock, clearBlocks } from '../../src/p2p/dht'

describe('DHT block announcements', () => {
  let nodeA: CoralNode
  let nodeB: CoralNode

  beforeAll(async () => {
    nodeA = await createCoralNode()
    nodeB = await createCoralNode()
    // Connect A ↔ B so they share a routing table
    await nodeA.libp2p.dial(nodeB.libp2p.getMultiaddrs()[0])
    // Give DHT time to bootstrap (exchange routing tables)
    await new Promise(r => setTimeout(r, 300))
  })

  afterAll(async () => {
    await nodeA.stop()
    await nodeB.stop()
  })

  it('announceBlocks() resolves without error', async () => {
    await expect(announceBlocks(nodeA.libp2p, 0, 7)).resolves.toBeUndefined()
  })

  it('findPeerForBlock() returns the announcing peer', async () => {
    await announceBlocks(nodeA.libp2p, 8, 15)
    // Give provider record time to propagate
    await new Promise(r => setTimeout(r, 200))
    const peers = await findPeerForBlock(nodeB.libp2p, 10)
    const peerIds = peers.map(p => p.peerId)
    expect(peerIds).toContain(nodeA.peerId)
  })

  it('findPeerForBlock() returns empty for unannounced block', async () => {
    const peers = await findPeerForBlock(nodeB.libp2p, 100)
    expect(peers).toHaveLength(0)
  })
})
