import type { OpenCoralNode } from './node'

export interface PeerModelInfo {
  repoId: string
  hfFilename: string
  blockStart: number
  blockEnd: number
  totalBlocks: number
  architecture: string
}

export interface NetworkPeer {
  peerId: string
  /** Display name shown in the UI (e.g. 'Coral Network for bootstrap nodes) */
  displayName?: string
  peerType: 'local' | 'remote' | 'discovery'
  multiaddrs: string[]
  blockRanges: { start: number; end: number }[]
  isLocal: boolean
  connected: boolean
  modelInfo?: PeerModelInfo
}

export interface NetworkConnection {
  from: string
  to: string
}

export interface NetworkState {
  localPeerId: string
  localMultiaddrs: string[]
  localBlocks: { start: number; end: number }[]
  peers: NetworkPeer[]
  connections: NetworkConnection[]
  timestamp: number
}

/**
 * Collects a snapshot of the network state visible from this node.
 * Shows the local node, all connected peers, and announced block ranges.
 */
export function inspectNetwork(
  node: OpenCoralNode,
  localBlocks: { start: number; end: number }[] = [],
  peerModelMap: Map<string, PeerModelInfo> = new Map(),
  bootstrapPeerIds: Set<string> = new Set(),
): NetworkState {
  const libp2p = node.libp2p
  const localPeerId = node.peerId

  const connectedPeerIds = libp2p.getPeers()

  const localModelInfo = peerModelMap.get(localPeerId)

  const peers: NetworkPeer[] = [
    {
      peerId: localPeerId,
      peerType: 'local',
      multiaddrs: node.multiaddrs,
      blockRanges: localBlocks,
      isLocal: true,
      connected: true,
      modelInfo: localModelInfo,
    },
  ]

  const connections: NetworkConnection[] = []

  for (const remotePeerId of connectedPeerIds) {
    const peerIdStr = remotePeerId.toString()
    const conns = libp2p.getConnections(remotePeerId)
    const addrs = conns.map(c => c.remoteAddr.toString())
    const remoteModelInfo = peerModelMap.get(peerIdStr)
    const isBootstrap = bootstrapPeerIds.has(peerIdStr)

    peers.push({
      peerId: peerIdStr,
      displayName: isBootstrap ? 'Coral Network' : undefined,
      peerType: isBootstrap ? 'discovery' : 'remote',
      multiaddrs: addrs,
      blockRanges: remoteModelInfo ? [{ start: remoteModelInfo.blockStart, end: remoteModelInfo.blockEnd }] : [],
      isLocal: false,
      connected: true,
      modelInfo: remoteModelInfo,
    })

    connections.push({
      from: localPeerId,
      to: peerIdStr,
    })
  }

  return {
    localPeerId,
    localMultiaddrs: node.multiaddrs,
    localBlocks,
    peers,
    connections,
    timestamp: Date.now(),
  }
}
