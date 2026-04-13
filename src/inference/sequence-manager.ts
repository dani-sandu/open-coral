import { peerIdFromString } from '@libp2p/peer-id'
import { multiaddr } from '@multiformats/multiaddr'
import type { OpenCoralNode } from '../p2p/node'
import { findPeerForBlock, type PeerBlockInfo } from '../p2p/dht'
import { sendInferenceRequest } from '../p2p/inference-protocol'
import { computeCoverage, type BlockCoverage, type CoverageReport } from './coverage'

/** Minimal interface for a block runner (sync BlockRunner or async AsyncBlockRunner). */
export interface BlockRunnerLike {
  readonly blockStart: number
  readonly blockEnd: number
  forward(input: Float32Array, nTokens: number): Float32Array | Promise<Float32Array>
}

export interface ChainStep {
  /** 'local' if handled by the local BlockRunner, otherwise a remote peer ID string */
  peerId: 'local' | string
  blockStart: number
  blockEnd: number
  /** First multiaddr of the remote peer — used if dialing is needed */
  multiaddr?: string
}

export interface ChainStepCandidate {
  peerId: string
  blockStart: number
  blockEnd: number
  multiaddr?: string
}

/**
 * A chain step with multiple peer candidates.
 * `runChainWithRetry()` tries them in order, moving to the next on transport error.
 */
export interface ChainStepWithCandidates {
  candidates: ChainStepCandidate[]
  blockStart: number
  blockEnd: number
}

export interface SequenceManagerOptions {
  node: OpenCoralNode
  /** Local BlockRunner, or null if this node is not hosting any blocks */
  localRunner: BlockRunnerLike | null
  totalBlocks: number
  hiddenSize: number
}

/**
 * Plans and executes a multi-hop inference chain across the P2P network.
 * Uses the Kademlia DHT to discover which peers host each block range.
 */
export class SequenceManager {
  private readonly node: OpenCoralNode
  private readonly localRunner: BlockRunnerLike | null
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
        current = await this.localRunner.forward(current, nTokens)
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
   * Build a chain where each step carries all known candidates for that block range.
   * Candidates are ordered best-first (first wins on retry). Use with runChainWithRetry().
   *
   * If `remoteCandidates` is provided, it is used directly (no DHT query).
   * Otherwise, the DHT is queried for all non-local blocks.
   */
  async planChainWithCandidates(
    remoteCandidates?: ChainStepCandidate[],
  ): Promise<ChainStepWithCandidates[]> {
    let blockCandidates: Map<number, ChainStepCandidate[]>

    if (remoteCandidates) {
      // Build candidate map from provided list
      blockCandidates = new Map()
      for (const c of remoteCandidates) {
        for (let i = c.blockStart; i <= c.blockEnd; i++) {
          const list = blockCandidates.get(i) ?? []
          list.push(c)
          blockCandidates.set(i, list)
        }
      }
    } else {
      // DHT-based: query each block index for all available peers
      blockCandidates = new Map()
      const selfId = this.node.peerId
      const queries: Promise<void>[] = []

      for (let i = 0; i < this.totalBlocks; i++) {
        if (this.localRunner && i >= this.localRunner.blockStart && i <= this.localRunner.blockEnd) continue
        queries.push(
          findPeerForBlock(this.node.libp2p, i)
            .then(peers => {
              const remote = peers.filter(p => p.peerId !== selfId)
              for (const peer of remote) {
                const list = blockCandidates.get(i) ?? []
                if (!list.some(c => c.peerId === peer.peerId)) {
                  list.push({
                    peerId: peer.peerId,
                    blockStart: i,
                    blockEnd: i,
                    multiaddr: peer.multiaddrs[0],
                  })
                }
                blockCandidates.set(i, list)
              }
            })
            .catch(() => {}),
        )
      }
      await Promise.all(queries)
    }

    const steps: ChainStepWithCandidates[] = []
    let position = 0

    while (position < this.totalBlocks) {
      // Local runner handles its range
      if (
        this.localRunner &&
        position >= this.localRunner.blockStart &&
        position <= this.localRunner.blockEnd
      ) {
        steps.push({
          candidates: [{ peerId: 'local', blockStart: this.localRunner.blockStart, blockEnd: this.localRunner.blockEnd }],
          blockStart: this.localRunner.blockStart,
          blockEnd: this.localRunner.blockEnd,
        })
        position = this.localRunner.blockEnd + 1
        continue
      }

      const candidates = blockCandidates.get(position)
      if (!candidates || candidates.length === 0) {
        throw new Error(`No peer found for blocks starting at ${position}`)
      }

      const best = candidates[0]
      let end = position
      // Extend range as long as the best peer covers consecutive blocks
      while (end + 1 < this.totalBlocks) {
        if (this.localRunner && end + 1 >= this.localRunner.blockStart && end + 1 <= this.localRunner.blockEnd) break
        if (!blockCandidates.get(end + 1)?.some(c => c.peerId === best.peerId)) break
        end++
      }

      steps.push({
        candidates,
        blockStart: position,
        blockEnd: end,
      })
      position = end + 1
    }

    return steps.sort((a, b) => a.blockStart - b.blockStart)
  }

  /**
   * Execute a chain with per-step retry. On transport error at any step,
   * the next candidate in that step's candidate list is tried.
   * Throws only if all candidates for a step fail.
   */
  async runChainWithRetry(
    chain: ChainStepWithCandidates[],
    input: Float32Array,
    nTokens: number,
  ): Promise<Float32Array> {
    let current = input

    for (const step of chain) {
      if (step.candidates[0].peerId === 'local') {
        if (!this.localRunner) throw new Error('SequenceManager: local step but no localRunner')
        current = this.localRunner.forward(current, nTokens)
        continue
      }

      let lastError: Error | null = null
      let succeeded = false

      for (const candidate of step.candidates) {
        try {
          const peerId = peerIdFromString(candidate.peerId)
          const isConnected = this.node.libp2p.getPeers().some(p => p.equals(peerId))
          if (!isConnected && candidate.multiaddr) {
            await this.node.libp2p.dial(multiaddr(candidate.multiaddr))
          }
          current = await sendInferenceRequest(
            this.node.libp2p,
            peerId,
            current,
            nTokens,
            this.hiddenSize,
          )
          succeeded = true
          break
        } catch (err) {
          lastError = err instanceof Error ? err : new Error(String(err))
          console.warn(`[SequenceManager] Peer ${candidate.peerId} failed (blocks ${step.blockStart}-${step.blockEnd}): ${lastError.message} — trying next candidate`)
        }
      }

      if (!succeeded) {
        throw lastError ?? new Error(`All candidates failed for blocks ${step.blockStart}-${step.blockEnd}`)
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
    const selfId = this.node.peerId

    for (let i = 0; i < this.totalBlocks; i++) {
      if (this.localRunner && i >= this.localRunner.blockStart && i <= this.localRunner.blockEnd) continue
      queries.push(
        findPeerForBlock(this.node.libp2p, i)
          .then(peers => {
            const remote = peers.filter(p => p.peerId !== selfId)
            if (remote.length > 0) blockPeers.set(i, remote[0])
          })
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
