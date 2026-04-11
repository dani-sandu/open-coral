import type { Libp2p } from 'libp2p'
import { announceBlocks as _announceBlocks } from './dht'

const DEFAULT_REANNOUNCE_INTERVAL_MS = 10 * 60 * 1000  // 10 minutes

export interface BlockRegistryOptions {
  blockStart: number
  blockEnd: number
  /** Override re-announce interval (ms). Default: 10 minutes. */
  reannounceIntervalMs?: number
  /** Override announceBlocks for testing. */
  announceBlocks?: (libp2p: Libp2p, start: number, end: number) => Promise<void>
}

/**
 * Manages this node's block range announcement in the Kademlia DHT.
 * Call start() after the libp2p node is running; call dispose() before stopping.
 */
export class BlockRegistry {
  private readonly libp2p: Libp2p
  private readonly blockStart: number
  private readonly blockEnd: number
  private readonly intervalMs: number
  private readonly announce: (libp2p: Libp2p, start: number, end: number) => Promise<void>
  private timer: ReturnType<typeof setInterval> | null = null

  constructor(libp2p: Libp2p, opts: BlockRegistryOptions) {
    this.libp2p = libp2p
    this.blockStart = opts.blockStart
    this.blockEnd = opts.blockEnd
    this.intervalMs = opts.reannounceIntervalMs ?? DEFAULT_REANNOUNCE_INTERVAL_MS
    this.announce = opts.announceBlocks ?? _announceBlocks
  }

  /** Announce immediately, then start periodic re-announcement. */
  async start(): Promise<void> {
    if (this.timer !== null) return  // already started
    await this.announce(this.libp2p, this.blockStart, this.blockEnd)
    this.timer = setInterval(() => {
      // Fire-and-forget re-announcement; errors are non-fatal
      this.announce(this.libp2p, this.blockStart, this.blockEnd).catch(err => {
        console.error(`[BlockRegistry] re-announce failed (blocks ${this.blockStart}-${this.blockEnd}):`, err)
      })
    }, this.intervalMs)
  }

  /** Cancel re-announcement timer. Does not withdraw the DHT record (it will expire). */
  dispose(): void {
    if (this.timer !== null) {
      clearInterval(this.timer)
      this.timer = null
    }
  }
}
