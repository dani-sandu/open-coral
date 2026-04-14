import { createLibp2p } from 'libp2p'
import { tcp } from '@libp2p/tcp'
import { noise, pureJsCrypto } from '@chainsafe/libp2p-noise'
import { yamux } from '@chainsafe/libp2p-yamux'
import { kadDHT, passthroughMapper } from '@libp2p/kad-dht'
import { identify } from '@libp2p/identify'
import { ping } from '@libp2p/ping'
import { generateKeyPair, privateKeyToProtobuf, privateKeyFromProtobuf } from '@libp2p/crypto/keys'
import { CID } from 'multiformats/cid'
import { sha256 } from 'multiformats/hashes/sha2'
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs'

// ── Identity persistence ─────────────────────────────────────────────────────

const DATA_DIR = process.env['DATA_DIR'] ?? '/data'
const IDENTITY_PATH = `${DATA_DIR}/identity.json`

async function loadOrCreatePrivateKey() {
  if (existsSync(IDENTITY_PATH)) {
    const stored = JSON.parse(readFileSync(IDENTITY_PATH, 'utf-8'))
    return privateKeyFromProtobuf(Buffer.from(stored.privKeyProto, 'base64'))
  }

  const privateKey = await generateKeyPair('Ed25519')
  const proto = privateKeyToProtobuf(privateKey)

  if (!existsSync(DATA_DIR)) mkdirSync(DATA_DIR, { recursive: true })
  writeFileSync(IDENTITY_PATH, JSON.stringify({
    privKeyProto: Buffer.from(proto).toString('base64'),
  }, null, 2), 'utf-8')

  return privateKey
}

// ── Presence announcement ────────────────────────────────────────────────────

const RAW = 0x55

async function coralNetworkCID(): Promise<CID> {
  const key = new TextEncoder().encode('coral/network/v1')
  const hash = await sha256.digest(key)
  return CID.createV1(RAW, hash)
}

async function announcePresence(libp2p: Awaited<ReturnType<typeof createLibp2p>>) {
  const cid = await coralNetworkCID()
  const signal = AbortSignal.timeout(5000)
  try {
    await libp2p.contentRouting.provide(cid, { signal })
  } catch (err: unknown) {
    if (!signal.aborted) throw err
  }
}

// ── Main ─────────────────────────────────────────────────────────────────────

const PORT = Number(process.env['PORT'] ?? 4001)

async function main() {
  const privateKey = await loadOrCreatePrivateKey()

  const node = await createLibp2p({
    privateKey,
    addresses: {
      listen: [`/ip4/0.0.0.0/tcp/${PORT}`],
    },
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
  })

  await node.start()

  const peerId = node.peerId.toString()
  const multiaddrs = node.getMultiaddrs().map(a => a.toString())

  console.log(`[bootstrap] Peer ID: ${peerId}`)
  for (const ma of multiaddrs) {
    console.log(`[bootstrap] Listening: ${ma}`)
  }

  // Announce presence immediately, then every 10 minutes
  await announcePresence(node)
  const timer = setInterval(() => {
    announcePresence(node).catch(err => {
      console.error('[bootstrap] re-announce failed:', err)
    })
  }, 10 * 60 * 1000)

  // Graceful shutdown
  const shutdown = async () => {
    console.log('[bootstrap] Shutting down...')
    clearInterval(timer)
    await node.stop()
    process.exit(0)
  }

  process.on('SIGTERM', shutdown)
  process.on('SIGINT', shutdown)
}

main().catch(err => {
  console.error('[bootstrap] Fatal:', err)
  process.exit(1)
})
