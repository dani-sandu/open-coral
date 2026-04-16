import { createLibp2p, type Libp2p } from 'libp2p'
import { tcp } from '@libp2p/tcp'
import { noise, pureJsCrypto } from '@chainsafe/libp2p-noise'
import { yamux } from '@chainsafe/libp2p-yamux'
import { kadDHT, passthroughMapper } from '@libp2p/kad-dht'
import { identify } from '@libp2p/identify'
import { ping } from '@libp2p/ping'
import { bootstrap } from '@libp2p/bootstrap'

export interface OpenCoralNode {
  /** base58btc-encoded peer ID */
  readonly peerId: string
  /** listening multiaddrs as strings, e.g. ['/ip4/127.0.0.1/tcp/54321'] */
  readonly multiaddrs: string[]
  /** the underlying libp2p instance — needed by dht.ts and inference-protocol.ts */
  readonly libp2p: Libp2p
  stop(): Promise<void>
}

export interface OpenCoralNodeOptions {
  /** TCP port to listen on. 0 = OS-assigned (default). */
  port?: number
  /** Override bootstrap multiaddrs. Empty array disables bootstrap. */
  bootstrapPeers?: string[]
}

/** Default bootstrap multiaddrs. Set OPENCORAL_BOOTSTRAP env var to override (comma-separated). */
export function getBootstrapPeers(overrides?: string[]): string[] {
  if (overrides !== undefined) return overrides
  const envVal = process.env['OPENCORAL_BOOTSTRAP']
  if (envVal) return envVal.split(',').map(s => s.trim()).filter(Boolean)
  return ['/ip4/72.62.154.149/tcp/4001/p2p/12D3KooWC33w9pdLEz65mrRain3EiMHe5EAxagCXSGoDjKLT4Q6J']
}

export async function createOpenCoralNode(opts: OpenCoralNodeOptions = {}): Promise<OpenCoralNode> {
  const port = opts.port ?? 0
  const peers = getBootstrapPeers(opts.bootstrapPeers)

  const libp2p = await createLibp2p({
    addresses: {
      listen: [`/ip4/0.0.0.0/tcp/${port}`],
    },
    transports: [tcp()],
    // pureJsCrypto uses @noble/ciphers (pure JS) for ChaCha20-Poly1305 instead of
    // Node.js createCipheriv, which Bun and Electron both handle incorrectly for
    // this cipher. Performance is identical in practice for our tensor payload sizes.
    connectionEncrypters: [noise({ crypto: pureJsCrypto })],
    streamMuxers: [yamux()],
    services: {
      identify: identify(),
      ping: ping(),
      dht: kadDHT({
        clientMode: false,
        protocol: '/opencoral/kad/1.0.0',
        // Keep private (127.x) addresses so local test peers enter the routing table.
        // The default removePrivateAddressesMapper strips them, leaving the table empty.
        peerInfoMapper: passthroughMapper,
      }),
    },
    ...(peers.length > 0 && {
      peerDiscovery: [
        bootstrap({ list: peers }),
      ],
    }),
  })

  await libp2p.start()

  return {
    get peerId() { return libp2p.peerId.toString() },
    get multiaddrs() { return libp2p.getMultiaddrs().map(a => a.toString()) },
    get libp2p() { return libp2p },
    stop: async () => { await libp2p.stop() },
  }
}
