import type { Libp2p } from 'libp2p'
import { CID } from 'multiformats/cid'
import { sha256 } from 'multiformats/hashes/sha2'

// RAW codec (0x55) — we're just hashing a string, not wrapping real content
const RAW = 0x55

export interface PeerBlockInfo {
  peerId: string
  multiaddrs: string[]
}

/** Deterministic CID for a block range, used as the DHT content-routing key. */
async function blockRangeCID(start: number, end: number): Promise<CID> {
  const key = new TextEncoder().encode(`coral/blocks/${start}-${end}`)
  const hash = await sha256.digest(key)
  return CID.createV1(RAW, hash)
}

/**
 * Announce to the DHT that this node hosts transformer blocks [start, end].
 * Other nodes can discover this peer by querying the same range.
 *
 * Best-effort: the provider record is stored locally even if the DHT query
 * times out before reaching remote peers (common in small networks).
 */
export async function announceBlocks(
  libp2p: Libp2p,
  start: number,
  end: number,
): Promise<void> {
  const cid = await blockRangeCID(start, end)
  const signal = AbortSignal.timeout(5000)
  try {
    await libp2p.contentRouting.provide(cid, { signal })
  } catch (err: unknown) {
    // signal.aborted = our own timeout (record still stored locally) — OK
    if (!signal.aborted) throw err
  }
}

/**
 * Find peers in the DHT that have announced hosting blocks [start, end].
 * Returns up to 20 results. Times out after 5 seconds with empty array.
 */
export async function findPeersForBlocks(
  libp2p: Libp2p,
  start: number,
  end: number,
): Promise<PeerBlockInfo[]> {
  const cid = await blockRangeCID(start, end)
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
    // signal.aborted is true when our own timeout fired (TimeoutError) — return partial results
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
