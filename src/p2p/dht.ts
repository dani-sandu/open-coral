import type { Libp2p } from 'libp2p'
import { CID } from 'multiformats/cid'
import { sha256 } from 'multiformats/hashes/sha2'

// RAW codec (0x55) — we're just hashing a string, not wrapping real content
const RAW = 0x55

export interface PeerBlockInfo {
  peerId: string
  multiaddrs: string[]
}

/** Deterministic CID for a model, used as the DHT content-routing key. */
async function modelCID(repoId: string): Promise<CID> {
  const key = new TextEncoder().encode(`coral/model/${repoId}/v1`)
  const hash = await sha256.digest(key)
  return CID.createV1(RAW, hash)
}

export async function announceModel(libp2p: Libp2p, repoId: string): Promise<void> {
  const cid = await modelCID(repoId)
  const signal = AbortSignal.timeout(5000)
  try {
    await libp2p.contentRouting.provide(cid, { signal })
  } catch (err: unknown) {
    if (!signal.aborted) throw err
  }
}

export async function findModelPeers(libp2p: Libp2p, repoId: string): Promise<PeerBlockInfo[]> {
  const cid = await modelCID(repoId)
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
