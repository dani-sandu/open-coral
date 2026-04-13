import { describe, it, expect, beforeAll, afterAll } from 'bun:test'
import { createOpenCoralNode, type OpenCoralNode } from '../../src/p2p/node'
import { announceBlocks, findPeerForBlock, clearBlocks, announcePresence, findCoralPeers } from '../../src/p2p/dht'

describe('DHT block announcements', () => {
  let nodeA: OpenCoralNode
  let nodeB: OpenCoralNode

  beforeAll(async () => {
    nodeA = await createOpenCoralNode()
    nodeB = await createOpenCoralNode()
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

describe('DHT presence announcement (single-record)', () => {
  let nodeC: OpenCoralNode
  let nodeD: OpenCoralNode

  beforeAll(async () => {
    nodeC = await createOpenCoralNode()
    nodeD = await createOpenCoralNode()
    await nodeC.libp2p.dial(nodeD.libp2p.getMultiaddrs()[0])
    await new Promise(r => setTimeout(r, 300))
  })

  afterAll(async () => {
    await nodeC.stop()
    await nodeD.stop()
  })

  it('announcePresence() resolves without error', async () => {
    await expect(announcePresence(nodeC.libp2p)).resolves.toBeUndefined()
  })

  it('findCoralPeers() returns the announcing peer', async () => {
    await announcePresence(nodeC.libp2p)
    await new Promise(r => setTimeout(r, 200))
    const peers = await findCoralPeers(nodeD.libp2p)
    const ids = peers.map(p => p.peerId)
    expect(ids).toContain(nodeC.peerId)
  })

  it('findCoralPeers() returns empty when no peer has announced', async () => {
    // Use fresh isolated nodes with no announcements
    const e = await createOpenCoralNode()
    const f = await createOpenCoralNode()
    await e.libp2p.dial(f.libp2p.getMultiaddrs()[0])
    await new Promise(r => setTimeout(r, 200))
    const peers = await findCoralPeers(f.libp2p)
    expect(peers).toHaveLength(0)
    await e.stop()
    await f.stop()
  })
})
