import { describe, it, expect, afterAll } from 'bun:test'
import { createLibp2p } from 'libp2p'
import { tcp } from '@libp2p/tcp'
import { noise, pureJsCrypto } from '@chainsafe/libp2p-noise'
import { yamux } from '@chainsafe/libp2p-yamux'
import { kadDHT, passthroughMapper } from '@libp2p/kad-dht'
import { identify } from '@libp2p/identify'
import { ping } from '@libp2p/ping'
import { bootstrap } from '@libp2p/bootstrap'
import { CID } from 'multiformats/cid'
import { sha256 } from 'multiformats/hashes/sha2'

const RAW = 0x55

async function coralNetworkCID(): Promise<CID> {
  const key = new TextEncoder().encode('coral/network/v1')
  const hash = await sha256.digest(key)
  return CID.createV1(RAW, hash)
}

function createNode(bootstrapAddrs: string[] = []) {
  return createLibp2p({
    addresses: { listen: ['/ip4/127.0.0.1/tcp/0'] },
    transports: [tcp()],
    connectionEncrypters: [noise({ crypto: pureJsCrypto })],
    streamMuxers: [yamux()],
    services: {
      identify: identify(),
      ping: ping(),
      dht: kadDHT({
        clientMode: false,
        protocol: '/opencoral/kad/1.0.0',
        peerInfoMapper: passthroughMapper,
      }),
    },
    ...(bootstrapAddrs.length > 0 && {
      peerDiscovery: [bootstrap({ list: bootstrapAddrs })],
    }),
  })
}

describe('Bootstrap-mediated peer discovery', () => {
  const nodes: Awaited<ReturnType<typeof createNode>>[] = []

  afterAll(async () => {
    await Promise.all(nodes.map(n => n.stop()))
  })

  it('two clients discover each other via a bootstrap node', async () => {
    // 1. Start the bootstrap node
    const bootstrapNode = await createNode()
    await bootstrapNode.start()
    nodes.push(bootstrapNode)

    const bootstrapAddr = bootstrapNode.getMultiaddrs()[0].toString()

    // 2. Start two client nodes that only know the bootstrap node
    const clientA = await createNode([bootstrapAddr])
    await clientA.start()
    nodes.push(clientA)

    const clientB = await createNode([bootstrapAddr])
    await clientB.start()
    nodes.push(clientB)

    // 3. Wait for clients to connect to the bootstrap node
    await new Promise(r => setTimeout(r, 1000))

    // 4. Client A announces presence
    const cid = await coralNetworkCID()
    await clientA.contentRouting.provide(cid, { signal: AbortSignal.timeout(5000) })

    // 5. Wait for DHT propagation
    await new Promise(r => setTimeout(r, 500))

    // 6. Client B discovers Client A
    const found: string[] = []
    const signal = AbortSignal.timeout(5000)
    try {
      for await (const provider of clientB.contentRouting.findProviders(cid, { signal })) {
        found.push(provider.id.toString())
        if (found.length >= 5) break
      }
    } catch {
      // timeout is acceptable — check what we found
    }

    expect(found).toContain(clientA.peerId.toString())
  }, 15_000)
})
