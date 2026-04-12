import { peerIdFromString } from '@libp2p/peer-id'
import { multiaddr } from '@multiformats/multiaddr'
import type { CoralNode } from '../p2p/node'
import type { BlockRunner } from './block-runner'
import { findPeerForBlock, type PeerBlockInfo } from '../p2p/dht'
import { sendInferenceRequest } from '../p2p/inference-protocol'
import { computeCoverage, type BlockCoverage, type CoverageReport } from './coverage'

export interface ChainStep {
  /** 'local' if handled by the local BlockRunner, otherwise a remote peer ID string */
  peerId: 'local' | string
  blockStart: number
  blockEnd: number
  /** First multiaddr of the remote peer — used if dialing is needed */
  multiaddr?: string
}

export interface SequenceManagerOptions {
  node: CoralNode
  /** Local BlockRunner, or null if this node is not hosting any blocks */
  localRunner: BlockRunner | null
  totalBlocks: number
  hiddenSize: number
}

/**
 * Plans and executes a multi-hop inference chain across the P2P network.
 * Uses the Kademlia DHT to discover which peers host each block range.
 */
export class SequenceManager {
  private readonly node: CoralNode
  private readonly localRunner: BlockRunner | null
  private readonly totalBlocks: number
  private readonly hiddenSize: number

  constructor(opts: SequenceManagerOptions) {
    this.node = opts.node
    this.localRunner = opts.localRunner
    this.totalBlocks = opts.totalBlocks
    this.hiddenSize = opts.hiddenSize
  }

  /**
   * Find a set of ChainSteps that together cover blocks 0..(totalBlocks-1).
   * Prefers the local runner for the ranges it hosts.
   * For uncovered ranges, queries the DHT (all blocks in parallel).
   * Throws if full coverage cannot be assembled.
   */
  async planChain(): Promise<ChainStep[]> {
    // Discover which peer hosts each non-local block, all in parallel
    const blockPeers = await this.discoverRemoteBlocks()

    const steps: ChainStep[] = []
    let position = 0

    while (position < this.totalBlocks) {
      // Check if local runner covers this position
      if (
        this.localRunner &&
        position >= this.localRunner.blockStart &&
        position <= this.localRunner.blockEnd
      ) {
        steps.push({
          peerId: 'local',
          blockStart: this.localRunner.blockStart,
          blockEnd: this.localRunner.blockEnd,
        })
        position = this.localRunner.blockEnd + 1
        continue
      }

      // Build a remote step from contiguous blocks hosted by the same peer
      const peer = blockPeers.get(position)
      if (!peer) {
        throw new Error(`No peer found for blocks starting at ${position} (need 0..${this.totalBlocks - 1})`)
      }

      let end = position
      while (end + 1 < this.totalBlocks) {
        // Stop at local runner boundary
        if (this.localRunner && end + 1 >= this.localRunner.blockStart && end + 1 <= this.localRunner.blockEnd) break
        const nextPeer = blockPeers.get(end + 1)
        if (!nextPeer || nextPeer.peerId !== peer.peerId) break
        end++
      }

      steps.push({
        peerId: peer.peerId,
        blockStart: position,
        blockEnd: end,
        multiaddr: peer.multiaddrs[0],
      })
      position = end + 1
    }

    return steps.sort((a, b) => a.blockStart - b.blockStart)
  }

  /**
   * Execute a chain of inference steps, passing hidden states from one peer to the next.
   *
   * @param chain   Steps from planChain()
   * @param input   Float32Array of shape [nTokens × hiddenSize]
   * @param nTokens Number of tokens in the batch
   * @returns       Float32Array of shape [nTokens × hiddenSize] after all blocks
   */
  async runChain(chain: ChainStep[], input: Float32Array, nTokens: number): Promise<Float32Array> {
    let current = input

    for (const step of chain) {
      if (step.peerId === 'local') {
        if (!this.localRunner) throw new Error('SequenceManager: local step but no localRunner')
        current = this.localRunner.forward(current, nTokens)
      } else {
        const peerId = peerIdFromString(step.peerId)
        const isConnected = this.node.libp2p.getPeers().some(p => p.equals(peerId))
        if (!isConnected && step.multiaddr) {
          await this.node.libp2p.dial(multiaddr(step.multiaddr))
        }
        current = await sendInferenceRequest(
          this.node.libp2p,
          peerId,
          current,
          nTokens,
          this.hiddenSize,
        )
      }
    }

    return current
  }

  /**
   * Query the DHT for every non-local block index in parallel.
   * Returns a map from block index → first responding peer.
   */
  private async discoverRemoteBlocks(): Promise<Map<number, PeerBlockInfo>> {
    const blockPeers = new Map<number, PeerBlockInfo>()
    const queries: Promise<void>[] = []

    for (let i = 0; i < this.totalBlocks; i++) {
      if (this.localRunner && i >= this.localRunner.blockStart && i <= this.localRunner.blockEnd) continue
      queries.push(
        findPeerForBlock(this.node.libp2p, i)
          .then(peers => { if (peers.length > 0) blockPeers.set(i, peers[0]) })
          .catch(() => {}),
      )
    }

    await Promise.all(queries)
    return blockPeers
  }

  /**
   * Non-throwing coverage check: queries the DHT for all block positions
   * in parallel and returns a CoverageReport describing which blocks are
   * covered and which are missing.
   */
  async checkCoverage(): Promise<CoverageReport> {
    const available: BlockCoverage[] = []

    // Include local runner upfront so computeCoverage can account for it
    if (this.localRunner) {
      available.push({
        blockStart: this.localRunner.blockStart,
        blockEnd: this.localRunner.blockEnd,
        peerId: 'local',
        multiaddrs: [],
      })
    }

    const blockPeers = await this.discoverRemoteBlocks()

    // Group contiguous blocks from the same peer into BlockCoverage ranges
    const sorted = [...blockPeers.entries()].sort((a, b) => a[0] - b[0])
    for (const [index, peer] of sorted) {
      const last = available[available.length - 1]
      if (last && last.peerId === peer.peerId && last.blockEnd === index - 1) {
        last.blockEnd = index
      } else {
        available.push({
          blockStart: index,
          blockEnd: index,
          peerId: peer.peerId,
          multiaddrs: peer.multiaddrs,
        })
      }
    }

    return computeCoverage(available, this.totalBlocks)
  }
}
