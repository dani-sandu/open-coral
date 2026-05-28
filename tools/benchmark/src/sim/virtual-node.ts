import { generateKeyPair } from '@libp2p/crypto/keys'
import { peerIdFromPrivateKey } from '@libp2p/peer-id'
import type { PeerId } from '@libp2p/interface'
import type { OpenCoralNode } from '../../../../src/p2p/node'
import { SimStream } from './sim-stream'

type StreamHandler = (stream: SimStream) => void | Promise<void>

/**
 * Minimal OpenCoralNode whose libp2p surface is backed by a SimNetwork.
 * Implements exactly the methods SequenceManager.runChain + sendInferenceRequestV3 touch.
 */
export class VirtualNode implements OpenCoralNode {
  readonly peerIdObj: PeerId
  readonly handlers = new Map<string, StreamHandler>()
  readonly libp2p: any

  private constructor(
    peerIdObj: PeerId,
    private readonly net: { peers(): VirtualNode[]; dialProtocol(target: PeerId, protocol: string): Promise<SimStream> },
  ) {
    this.peerIdObj = peerIdObj
    const self = this
    this.libp2p = {
      peerId: peerIdObj,
      getPeers(): PeerId[] {
        // Report every other node as connected so runChain skips dialing.
        return self.net.peers().filter(p => p !== self).map(p => p.peerIdObj)
      },
      getConnections(): unknown[] { return [] },
      async dial(): Promise<void> { /* no-op: in-process, already "connected" */ },
      async dialProtocol(target: PeerId, protocol: string): Promise<SimStream> {
        return self.net.dialProtocol(target, protocol)
      },
      async handle(protocol: string, handler: StreamHandler, _opts?: unknown): Promise<void> {
        self.handlers.set(protocol, handler)
      },
      async unhandle(protocol: string): Promise<void> { self.handlers.delete(protocol) },
    }
  }

  static async create(net: { peers(): VirtualNode[]; dialProtocol(target: PeerId, protocol: string): Promise<SimStream> }): Promise<VirtualNode> {
    const key = await generateKeyPair('Ed25519')
    const peerIdObj = peerIdFromPrivateKey(key)
    return new VirtualNode(peerIdObj, net)
  }

  get peerId(): string { return this.peerIdObj.toString() }
  get multiaddrs(): string[] { return [`/ip4/127.0.0.1/tcp/0/p2p/${this.peerId}`] }
  async stop(): Promise<void> { this.handlers.clear() }
}
