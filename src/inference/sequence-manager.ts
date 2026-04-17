import { peerIdFromString } from '@libp2p/peer-id'
import { multiaddr } from '@multiformats/multiaddr'
import type { OpenCoralNode } from '../p2p/node'
import { findCoralPeers, findModelPeers, type PeerBlockInfo } from '../p2p/dht'
import { queryPeerModelInfo } from '../p2p/model-announce'
import type { PeerLatencyTracker } from '../p2p/peer-latency'
import { DEFAULT_LATENCY_MS } from '../p2p/peer-latency'
import { sendInferenceRequestV3 } from '../p2p/inference-protocol'
import type { NodeIdentity } from '../main/identity'
import { computeCoverage, type BlockCoverage, type CoverageReport } from './coverage'

const HOP_PENALTY_MS = 50

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
  /**
   * Callback to look up the block range a peer is hosting.
   * Returns null if the peer's range is unknown (not yet queried via modelinfo).
   */
  getPeerBlockRange?: (peerId: string) => { blockStart: number; blockEnd: number } | null
  /** Optional tracker for latency-aware peer selection. */
  latencyTracker?: PeerLatencyTracker
  /** Node identity used to sign outgoing V3 inference requests. */
  identity: NodeIdentity
  /** Model repo ID used for model-level DHT discovery (optional). */
  repoId?: string
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
  private readonly getPeerBlockRange: (peerId: string) => { blockStart: number; blockEnd: number } | null
  private readonly latencyTracker: PeerLatencyTracker | null
  private readonly identity: NodeIdentity
  private readonly repoId: string | undefined

  constructor(opts: SequenceManagerOptions) {
    this.node = opts.node
    this.localRunner = opts.localRunner
    this.totalBlocks = opts.totalBlocks
    this.hiddenSize = opts.hiddenSize
    this.getPeerBlockRange = opts.getPeerBlockRange ?? (() => null)
    this.latencyTracker = opts.latencyTracker ?? null
    this.identity = opts.identity
    this.repoId = opts.repoId
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
  async runChain(chain: ChainStep[], input: Float32Array, nTokens: number, requestId = 'no-trace'): Promise<Float32Array> {
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
        const t0 = Date.now()
        current = await sendInferenceRequestV3(
          this.node.libp2p,
          peerId,
          current,
          nTokens,
          this.hiddenSize,
          requestId,
          this.identity,
        )
        this.latencyTracker?.record(step.peerId, Date.now() - t0)
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
      blockCandidates = new Map()
      const selfId = this.node.peerId

      // Helper: query a peer for model info and add to candidates
      const addPeerCandidates = async (peerId: string, multiaddrs: string[]): Promise<void> => {
        if (peerId === selfId) return
        try {
          const pid = peerIdFromString(peerId)
          const info = await queryPeerModelInfo(this.node.libp2p, pid)
          if (!info) return
          for (let i = info.blockStart; i <= info.blockEnd; i++) {
            if (this.localRunner && i >= this.localRunner.blockStart && i <= this.localRunner.blockEnd) continue
            const list = blockCandidates.get(i) ?? []
            if (!list.some(c => c.peerId === peerId)) {
              list.push({
                peerId,
                blockStart: info.blockStart,
                blockEnd: info.blockEnd,
                multiaddr: multiaddrs[0],
              })
            }
            blockCandidates.set(i, list)
          }
        } catch {}
      }

      if (this.repoId) {
        // Model-level discovery: DHT query, then modelinfo per peer
        const peers = await findModelPeers(this.node.libp2p, this.repoId)
        for (const peer of peers) {
          await addPeerCandidates(peer.peerId, peer.multiaddrs)
        }

        // Fallback: if DHT returned no usable results (e.g. bootstrap disconnected),
        // query all directly connected peers — they're still reachable even without DHT.
        if (blockCandidates.size === 0) {
          const connectedPeers = this.node.libp2p.getPeers()
          for (const pid of connectedPeers) {
            const peerIdStr = pid.toString()
            const addrs = this.node.libp2p.getConnections(pid).flatMap(c => [c.remoteAddr.toString()])
            await addPeerCandidates(peerIdStr, addrs)
          }
        }
      } else {
        // Fallback: coral peer discovery + getPeerBlockRange
        const coralPeers = await findCoralPeers(this.node.libp2p)
        for (const peer of coralPeers) {
          if (peer.peerId === selfId) continue
          const range = this.getPeerBlockRange(peer.peerId)
          if (!range) continue
          for (let i = range.blockStart; i <= range.blockEnd; i++) {
            if (this.localRunner && i >= this.localRunner.blockStart && i <= this.localRunner.blockEnd) continue
            const list = blockCandidates.get(i) ?? []
            if (!list.some(c => c.peerId === peer.peerId)) {
              list.push({
                peerId: peer.peerId,
                blockStart: range.blockStart,
                blockEnd: range.blockEnd,
                multiaddr: peer.multiaddrs[0],
              })
            }
            blockCandidates.set(i, list)
          }
        }
      }
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

      // Score candidates by latency, with a bonus for candidates that cover more
      // remaining blocks (avoids picking a slightly-faster peer that forces a hop soon)
      const remainingBlocks = this.totalBlocks - position
      const scored = candidates.map(c => {
        const latency = this.latencyTracker?.getEstimate(c.peerId) ?? DEFAULT_LATENCY_MS
        // How far can this candidate extend from the current position?
        let reach = 0
        for (let j = position; j < this.totalBlocks; j++) {
          if (blockCandidates.get(j)?.some(bc => bc.peerId === c.peerId)) reach++
          else break
        }
        // A candidate that can't cover all remaining blocks will force a hop later.
        // Add a hop penalty proportional to the fraction of range NOT covered.
        const coveragePenalty = reach < remainingBlocks ? HOP_PENALTY_MS : 0
        return { candidate: c, latency, effectiveScore: latency + coveragePenalty }
      }).sort((a, b) => a.effectiveScore - b.effectiveScore)

      const currentPeer = scored[0]
      let end = position

      while (end + 1 < this.totalBlocks) {
        if (this.localRunner && end + 1 >= this.localRunner.blockStart && end + 1 <= this.localRunner.blockEnd) break

        const nextCandidates = blockCandidates.get(end + 1)
        if (!nextCandidates || nextCandidates.length === 0) break

        // Can current peer continue?
        const canContinue = nextCandidates.some(c => c.peerId === currentPeer.candidate.peerId)
        if (!canContinue) break

        // Is there a better peer at the next position? (accounting for hop penalty)
        const continuationScore = currentPeer.latency
        const alternatives = nextCandidates
          .filter(c => c.peerId !== currentPeer.candidate.peerId)
          .map(c => ({
            candidate: c,
            latency: this.latencyTracker?.getEstimate(c.peerId) ?? DEFAULT_LATENCY_MS,
          }))

        const bestAlt = alternatives.length > 0
          ? alternatives.reduce((a, b) => a.latency < b.latency ? a : b)
          : null

        if (bestAlt && bestAlt.latency + HOP_PENALTY_MS < continuationScore) {
          break  // Split here — a significantly faster peer is available
        }

        end++
      }

      // Order all candidates for this range by latency (best-first)
      const stepCandidates = (blockCandidates.get(position) ?? [])
        .map(c => ({
          ...c,
          _latency: this.latencyTracker?.getEstimate(c.peerId) ?? DEFAULT_LATENCY_MS,
        }))
        .sort((a, b) => a._latency - b._latency)
        .map(({ _latency, ...c }) => c)

      steps.push({
        candidates: stepCandidates,
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
    requestId = 'no-trace',
  ): Promise<Float32Array> {
    let current = input

    for (const step of chain) {
      if (step.candidates[0].peerId === 'local') {
        if (!this.localRunner) throw new Error('SequenceManager: local step but no localRunner')
        current = await this.localRunner.forward(current, nTokens)
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
          const t0 = Date.now()
          current = await sendInferenceRequestV3(
            this.node.libp2p,
            peerId,
            current,
            nTokens,
            this.hiddenSize,
            requestId,
            this.identity,
          )
          this.latencyTracker?.record(candidate.peerId, Date.now() - t0)
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
   * Find coral peers via a single DHT lookup, map each to their block range
   * via the getPeerBlockRange callback, and build a block→best-peer map.
   * Prefers peers with lower latency when multiple cover the same block.
   */
  private async discoverRemoteBlocks(): Promise<Map<number, PeerBlockInfo>> {
    const coralPeers = await findCoralPeers(this.node.libp2p)
    const selfId = this.node.peerId

    // Map: blockIndex → list of candidate PeerBlockInfos
    const candidates = new Map<number, PeerBlockInfo[]>()

    for (const peer of coralPeers) {
      if (peer.peerId === selfId) continue

      const range = this.getPeerBlockRange(peer.peerId)
      if (!range) continue

      for (let i = range.blockStart; i <= range.blockEnd; i++) {
        // Skip blocks handled by local runner
        if (this.localRunner && i >= this.localRunner.blockStart && i <= this.localRunner.blockEnd) continue

        const list = candidates.get(i) ?? []
        list.push(peer)
        candidates.set(i, list)
      }
    }

    // For each block, pick the best (lowest latency) candidate
    const blockPeers = new Map<number, PeerBlockInfo>()
    for (const [index, peers] of candidates) {
      if (peers.length === 1) {
        blockPeers.set(index, peers[0])
      } else {
        const bestId = this.latencyTracker?.bestPeer(peers.map(p => p.peerId)) ?? peers[0].peerId
        const best = peers.find(p => p.peerId === bestId) ?? peers[0]
        blockPeers.set(index, best)
      }
    }

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

    // Actively query peers for their block ranges (same approach as planChainWithCandidates).
    // discoverRemoteBlocks() uses a stale cache — coverage must use fresh data.
    const selfId = this.node.peerId

    // Helper: query a peer and add to coverage
    const addPeerCoverage = async (peerId: string, multiaddrs: string[]): Promise<void> => {
      if (peerId === selfId) return
      try {
        const pid = peerIdFromString(peerId)
        const info = await queryPeerModelInfo(this.node.libp2p, pid)
        if (!info) return
        available.push({
          blockStart: info.blockStart,
          blockEnd: info.blockEnd,
          peerId,
          multiaddrs,
        })
      } catch {}
    }

    if (this.repoId) {
      const peers = await findModelPeers(this.node.libp2p, this.repoId)
      for (const peer of peers) {
        await addPeerCoverage(peer.peerId, peer.multiaddrs)
      }

      // Fallback: if DHT returned no remote peers, query directly connected peers
      const hasRemoteCoverage = available.some(a => a.peerId !== 'local')
      if (!hasRemoteCoverage) {
        const connectedPeers = this.node.libp2p.getPeers()
        for (const pid of connectedPeers) {
          const addrs = this.node.libp2p.getConnections(pid).flatMap(c => [c.remoteAddr.toString()])
          await addPeerCoverage(pid.toString(), addrs)
        }
      }
    } else {
      // Fallback: use cached discovery (same as before)
      const blockPeers = await this.discoverRemoteBlocks()
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
    }

    return computeCoverage(available, this.totalBlocks)
  }
}
