import { describe, it, expect, beforeAll, afterAll } from 'bun:test'
import { createOpenCoralNode, type OpenCoralNode } from '../../src/p2p/node'
import { announceModel, findModelPeers, announcePresence, findCoralPeers } from '../../src/p2p/dht'

describe('DHT model announcements', () => {
  let nodeA: OpenCoralNode
  let nodeB: OpenCoralNode

  beforeAll(async () => {
    nodeA = await createOpenCoralNode()
    nodeB = await createOpenCoralNode()
    await nodeA.libp2p.dial(nodeB.libp2p.getMultiaddrs()[0])
    await new Promise(r => setTimeout(r, 300))
  })

  afterAll(async () => {
    await nodeA.stop()
    await nodeB.stop()
  })

  it('announceModel() resolves without error', async () => {
    await expect(announceModel(nodeA.libp2p, 'test-org/test-model')).resolves.toBeUndefined()
  })

  it('findModelPeers() returns the announcing peer', async () => {
    await announceModel(nodeA.libp2p, 'test-org/model-abc')
    await new Promise(r => setTimeout(r, 200))
    const peers = await findModelPeers(nodeB.libp2p, 'test-org/model-abc')
    const peerIds = peers.map(p => p.peerId)
    expect(peerIds).toContain(nodeA.peerId)
  })

  it('findModelPeers() returns empty for unannounced model', async () => {
    const peers = await findModelPeers(nodeB.libp2p, 'nonexistent/model')
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
