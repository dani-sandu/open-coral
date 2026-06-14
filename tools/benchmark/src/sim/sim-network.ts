import type { PeerId } from '@libp2p/interface'
import { Channel } from './channel'
import { SimStream } from './sim-stream'
import { VirtualNode } from './virtual-node'
import { INFERENCE_PROTOCOL_V3 } from '../../../../src/p2p/inference-protocol'

export interface SimNetworkOptions {
  modelBlocks: number
  latencyMeanMs: number
  latencyJitterMs: number
  /** Opt-in: one-time handshake latency charged on first dial to a peer. 0 = peers pre-connected (default). */
  dialLatencyMs?: number
}

/**
 * Routes dialProtocol calls between VirtualNodes in-process and injects N(μ,σ)
 * latency at handler dispatch. No TCP, no noise handshake — only the transport
 * is mocked; SequenceManager / V3 protocol run unmodified.
 */
export class SimNetwork {
  private readonly nodes = new Map<string, VirtualNode>()
  private spare = 0 // cached second Box-Muller value
  private hasSpare = false

  constructor(private readonly opts: SimNetworkOptions) {}

  peers(): VirtualNode[] { return [...this.nodes.values()] }

  async addNode(): Promise<VirtualNode> {
    const node = await VirtualNode.create({
      peers: () => this.peers(),
      dialProtocol: (target, protocol) => this.dialProtocol(target, protocol),
      connect: (from, targetId) => this.connect(from, targetId),
      connectionModeling: () => this.connectionModeling(),
      getPeerIdObj: (id) => this.getPeerIdObj(id),
    })
    this.nodes.set(node.peerId, node)
    return node
  }

  /** True when connection modeling is active (dialLatencyMs > 0). */
  connectionModeling(): boolean {
    return (this.opts.dialLatencyMs ?? 0) > 0
  }

  sampleDialLatency(): number {
    return Math.max(0, this.opts.dialLatencyMs ?? 0)
  }

  /** Resolve a peerId string to its VirtualNode's PeerId object (for getPeers). */
  getPeerIdObj(peerId: string): import('@libp2p/interface').PeerId | undefined {
    return this.nodes.get(peerId)?.peerIdObj
  }

  /**
   * Establish a connection from `from` to `targetId`. No-op in auto-connected mode.
   * Idempotent: already-connected → instant; concurrent dials share one handshake.
   */
  async connect(from: VirtualNode, targetId: string): Promise<void> {
    if (!this.connectionModeling()) return
    if (from.connected.has(targetId)) return
    let inFlight = from.dialing.get(targetId)
    if (!inFlight) {
      inFlight = new Promise<void>(resolve => {
        setTimeout(() => {
          from.connected.add(targetId)
          from.dialing.delete(targetId)
          resolve()
        }, this.sampleDialLatency())
      })
      from.dialing.set(targetId, inFlight)
    }
    await inFlight
  }

  /** Draw a latency sample from N(mean, jitter), clamped to [0, mean + 5*jitter]. */
  sampleLatency(): number {
    const { latencyMeanMs: mean, latencyJitterMs: jitter } = this.opts
    if (jitter <= 0) return Math.max(0, mean)
    let z: number
    if (this.hasSpare) { z = this.spare; this.hasSpare = false }
    else {
      // Box-Muller
      let u = 0, v = 0
      while (u === 0) u = Math.random()
      while (v === 0) v = Math.random()
      const mag = Math.sqrt(-2 * Math.log(u))
      z = mag * Math.cos(2 * Math.PI * v)
      this.spare = mag * Math.sin(2 * Math.PI * v)
      this.hasSpare = true
    }
    const sample = mean + jitter * z
    return Math.min(Math.max(0, sample), mean + 5 * jitter)
  }

  /**
   * Create a duplex stream pair, hand the responder end to the target's
   * registered handler after a sampled latency delay, and return the
   * initiator end. Throws if the target has no handler (models a dead peer
   * → triggers SequenceManager retry).
   */
  async dialProtocol(target: PeerId, protocols: string | string[]): Promise<SimStream> {
    const node = this.nodes.get(target.toString())
    if (!node) throw new Error(`SimNetwork: no node for peer ${target.toString()}`)

    const wanted = Array.isArray(protocols) ? protocols : [protocols]
    const chosen = wanted.find(p => node.handlers.has(p))
    if (!chosen) {
      throw new Error(`SimNetwork: peer ${target.toString()} supports none of [${wanted.join(', ')}]`)
    }
    const handler = node.handlers.get(chosen)!

    const a2b = new Channel() // initiator → responder
    const b2a = new Channel() // responder → initiator
    const initiator = new SimStream(a2b, b2a)
    const responder = new SimStream(b2a, a2b)
    initiator.protocol = chosen
    responder.protocol = chosen

    const latency = this.sampleLatency()
    setTimeout(() => { void Promise.resolve(handler(responder)).catch(() => {}) }, latency)

    return initiator
  }

  /** Convenience re-export so callers don't import the protocol string separately. */
  get inferenceProtocol(): string { return INFERENCE_PROTOCOL_V3 }
}
