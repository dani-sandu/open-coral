import { Channel } from './channel'

/**
 * A libp2p-compatible duplex stream over two in-process Channels.
 * `write` carries bytes to the peer; `read` carries bytes from the peer.
 * Only the subset of the Stream API used by stream-utils.ts is implemented.
 */
export class SimStream implements AsyncIterable<Uint8Array> {
  /** Negotiated protocol id, set by SimNetwork.dialProtocol. */
  protocol = ''

  constructor(
    private readonly write: Channel,
    private readonly read: Channel,
  ) {}

  /** Always succeeds in-process — backpressure ('drain') is never triggered. */
  send(data: Uint8Array): boolean {
    this.write.push(data)
    return true
  }

  /** No-op: send() never returns false, so 'drain' listeners never fire. */
  addEventListener(_event: 'drain', _cb: () => void, _opts?: { once?: boolean }): void {}

  /** Close the outbound side so the peer's reader terminates. Reading is unaffected. */
  async close(): Promise<void> {
    this.write.close()
  }

  /** Tear down both directions. */
  abort(_err?: Error): void {
    this.write.close()
    this.read.close()
  }

  [Symbol.asyncIterator](): AsyncIterator<Uint8Array> {
    return this.read[Symbol.asyncIterator]()
  }
}
