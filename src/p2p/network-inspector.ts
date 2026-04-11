import type { CoralNode } from './node'

export interface NetworkPeer {
  peerId: string
  multiaddrs: string[]
  blockRanges: { start: number; end: number }[]
  isLocal: boolean
  connected: boolean
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
  node: CoralNode,
  localBlocks: { start: number; end: number }[] = [],
): NetworkState {
  const libp2p = node.libp2p
  const localPeerId = node.peerId

  const connectedPeerIds = libp2p.getPeers()

  const peers: NetworkPeer[] = [
    {
      peerId: localPeerId,
      multiaddrs: node.multiaddrs,
      blockRanges: localBlocks,
      isLocal: true,
      connected: true,
    },
  ]

  const connections: NetworkConnection[] = []

  for (const remotePeerId of connectedPeerIds) {
    const peerIdStr = remotePeerId.toString()
    const conns = libp2p.getConnections(remotePeerId)
    const addrs = conns.map(c => c.remoteAddr.toString())

    peers.push({
      peerId: peerIdStr,
      multiaddrs: addrs,
      blockRanges: [], // populated when peers announce via BlockRegistry
      isLocal: false,
      connected: true,
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
