import type { Libp2p } from 'libp2p'
import { CID } from 'multiformats/cid'
import { sha256 } from 'multiformats/hashes/sha2'

// RAW codec (0x55) — we're just hashing a string, not wrapping real content
const RAW = 0x55

export interface PeerBlockInfo {
  peerId: string
  multiaddrs: string[]
}

/** Deterministic CID for a single block index, used as the DHT content-routing key. */
async function singleBlockCID(index: number): Promise<CID> {
  const key = new TextEncoder().encode(`opencoral/block/${index}`)
  const hash = await sha256.digest(key)
  return CID.createV1(RAW, hash)
}

/**
 * Announce to the DHT that this node hosts transformer blocks [start, end].
 * Publishes a separate provider record for each block index so that
 * per-block lookups resolve in O(1) instead of requiring O(N²) range scans.
 *
 * Best-effort: provider records are stored locally even if the DHT query
 * times out before reaching remote peers (common in small networks).
 */
export async function announceBlocks(
  libp2p: Libp2p,
  start: number,
  end: number,
): Promise<void> {
  const promises: Promise<void>[] = []
  for (let i = start; i <= end; i++) {
    promises.push(
      (async () => {
        const cid = await singleBlockCID(i)
        const signal = AbortSignal.timeout(5000)
        try {
          await libp2p.contentRouting.provide(cid, { signal })
        } catch (err: unknown) {
          if (!signal.aborted) throw err
        }
      })(),
    )
  }
  await Promise.all(promises)
}

/**
 * Find peers in the DHT that host a specific block index.
 * Returns up to 20 results. Times out after 5 seconds with empty array.
 */
export async function findPeerForBlock(
  libp2p: Libp2p,
  index: number,
): Promise<PeerBlockInfo[]> {
  const cid = await singleBlockCID(index)
  const results: PeerBlockInfo[] = []
  const signal = AbortSignal.timeout(5000)
  try {
    for await (const provider of libp2p.contentRouting.findProviders(cid, { signal })) {
      results.push({
        peerId: provider.id.toString(),
        multiaddrs: provider.multiaddrs.map(a => a.toString()),
      })
      if (results.length >= 20) break
    }
  } catch (err: unknown) {
    if (!signal.aborted) throw err
  }
  return results
}

/**
 * Withdraw the announcement that this node hosts blocks [start, end].
 * Note: Kademlia provider records expire automatically after ~24 hours.
 * Not all libp2p versions support explicit record removal — this is best-effort.
 */
export async function clearBlocks(
  _libp2p: Libp2p,
  _start: number,
  _end: number,
): Promise<void> {
  // Provider records expire automatically via DHT TTL (~24h).
  // cancelReprovide() is available in kad-dht v16 but not yet wired up.
}

// ── Single-record presence announcement ───────────────────────────────────────

/** Deterministic CID representing the Coral P2P network. All Coral nodes provide
 *  this CID so peers can discover each other with a single DHT query.
 *  Cached after first computation — the key is static and the result never changes. */
let _coralNetworkCID: Promise<CID> | undefined
function coralNetworkCID(): Promise<CID> {
  if (!_coralNetworkCID) {
    _coralNetworkCID = (async () => {
      const key = new TextEncoder().encode('coral/network/v1')
      const hash = await sha256.digest(key)
      return CID.createV1(RAW, hash)
    })()
  }
  return _coralNetworkCID
}

/**
 * Announce this node's presence on the Coral network with a single DHT record.
 * Call this instead of (or in addition to) announceBlocks when hosting blocks.
 */
export async function announcePresence(libp2p: Libp2p): Promise<void> {
  const cid = await coralNetworkCID()
  const signal = AbortSignal.timeout(5000)
  try {
    await libp2p.contentRouting.provide(cid, { signal })
  } catch (err: unknown) {
    if (!signal.aborted) throw err
  }
}

/**
 * Find all Coral peers known to the DHT.
 * Returns up to 20 results. Times out after 5 seconds with empty array.
 */
export async function findCoralPeers(libp2p: Libp2p): Promise<PeerBlockInfo[]> {
  const cid = await coralNetworkCID()
  const results: PeerBlockInfo[] = []
  const signal = AbortSignal.timeout(5000)
  try {
    for await (const provider of libp2p.contentRouting.findProviders(cid, { signal })) {
      results.push({
        peerId: provider.id.toString(),
        multiaddrs: provider.multiaddrs.map(a => a.toString()),
      })
      if (results.length >= 20) break
    }
  } catch (err: unknown) {
    if (!signal.aborted) throw err
  }
  return results
}
