import { peerIdFromString } from '@libp2p/peer-id'
import { multiaddr } from '@multiformats/multiaddr'
import type { CoralNode } from '../p2p/node'
import type { BlockRunner } from './block-runner'
import { findPeersForBlocks } from '../p2p/dht'
import { sendInferenceRequest } from '../p2p/inference-protocol'

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
   * For uncovered ranges, queries the DHT.
   * Throws if full coverage cannot be assembled.
   */
  async planChain(): Promise<ChainStep[]> {
    const steps: ChainStep[] = []
    let covered = 0

    // Insert local runner coverage if it starts at or before `covered`
    if (this.localRunner && this.localRunner.blockStart === 0) {
      steps.push({
        peerId: 'local',
        blockStart: this.localRunner.blockStart,
        blockEnd: this.localRunner.blockEnd,
      })
      covered = this.localRunner.blockEnd + 1
    }

    while (covered < this.totalBlocks) {
      // Check if local runner covers this range (non-zero start)
      if (
        this.localRunner &&
        this.localRunner.blockStart === covered
      ) {
        steps.push({
          peerId: 'local',
          blockStart: this.localRunner.blockStart,
          blockEnd: this.localRunner.blockEnd,
        })
        covered = this.localRunner.blockEnd + 1
        continue
      }

      // Query DHT for ranges starting at `covered`, try largest first
      let found = false
      for (let end = this.totalBlocks - 1; end >= covered; end--) {
        const peers = await findPeersForBlocks(this.node.libp2p, covered, end)
        if (peers.length > 0) {
          const peer = peers[0]
          steps.push({
            peerId: peer.peerId,
            blockStart: covered,
            blockEnd: end,
            multiaddr: peer.multiaddrs[0],
          })
          covered = end + 1
          found = true
          break
        }
      }
      if (!found) {
        throw new Error(`No peer found for blocks starting at ${covered} (need 0..${this.totalBlocks - 1})`)
      }
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
}
