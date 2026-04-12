import { createLibp2p, type Libp2p } from 'libp2p'
import { tcp } from '@libp2p/tcp'
import { noise } from '@chainsafe/libp2p-noise'
import { yamux } from '@chainsafe/libp2p-yamux'
import { kadDHT, passthroughMapper } from '@libp2p/kad-dht'
import { identify } from '@libp2p/identify'
import { ping } from '@libp2p/ping'

export interface CoralNode {
  /** base58btc-encoded peer ID */
  readonly peerId: string
  /** listening multiaddrs as strings, e.g. ['/ip4/127.0.0.1/tcp/54321'] */
  readonly multiaddrs: string[]
  /** the underlying libp2p instance — needed by dht.ts and inference-protocol.ts */
  readonly libp2p: Libp2p
  stop(): Promise<void>
}

export interface CoralNodeOptions {
  /** TCP port to listen on. 0 = OS-assigned (default). */
  port?: number
}

export async function createCoralNode(opts: CoralNodeOptions = {}): Promise<CoralNode> {
  const port = opts.port ?? 0

  const libp2p = await createLibp2p({
    addresses: {
      listen: [`/ip4/0.0.0.0/tcp/${port}`],
    },
    transports: [tcp()],
    connectionEncrypters: [noise()],
    streamMuxers: [yamux()],
    services: {
      identify: identify(),
      ping: ping(),
      dht: kadDHT({
        clientMode: false,
        protocol: '/coral/kad/1.0.0',
        // Keep private (127.x) addresses so local test peers enter the routing table.
        // The default removePrivateAddressesMapper strips them, leaving the table empty.
        peerInfoMapper: passthroughMapper,
      }),
    },
  })

  await libp2p.start()

  return {
    get peerId() { return libp2p.peerId.toString() },
    get multiaddrs() { return libp2p.getMultiaddrs().map(a => a.toString()) },
    get libp2p() { return libp2p },
    stop: async () => { await libp2p.stop() },
  }
}
