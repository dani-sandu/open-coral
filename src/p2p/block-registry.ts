import type { Libp2p } from 'libp2p'
import { announcePresence, announceModel } from './dht'

const DEFAULT_REANNOUNCE_INTERVAL_MS = 10 * 60 * 1000  // 10 minutes

export interface BlockRegistryOptions {
  /** Override re-announce interval (ms). Default: 10 minutes. */
  reannounceIntervalMs?: number
  /** Override announcePresence for testing. */
  announcePresence?: (libp2p: Libp2p) => Promise<void>
  /** Override announceModel for testing. */
  announceModel?: (libp2p: Libp2p, repoId: string) => Promise<void>
}

/**
 * Manages this node's block range announcement in the Kademlia DHT.
 * Call start() after the libp2p node is running; call dispose() before stopping.
 */
export class BlockRegistry {
  private readonly libp2p: Libp2p
  private readonly repoId: string
  private readonly intervalMs: number
  private readonly announce: (libp2p: Libp2p) => Promise<void>
  private readonly announceModelFn: (libp2p: Libp2p, repoId: string) => Promise<void>
  private timer: ReturnType<typeof setInterval> | null = null

  constructor(libp2p: Libp2p, repoId: string, opts: BlockRegistryOptions) {
    this.libp2p = libp2p
    this.repoId = repoId
    this.intervalMs = opts.reannounceIntervalMs ?? DEFAULT_REANNOUNCE_INTERVAL_MS
    this.announce = opts.announcePresence ?? announcePresence
    this.announceModelFn = opts.announceModel ?? announceModel
  }

  /** Announce immediately, then start periodic re-announcement. */
  async start(): Promise<void> {
    if (this.timer !== null) return  // already started
    await this.announce(this.libp2p)
    await this.announceModelFn(this.libp2p, this.repoId)
    this.timer = setInterval(() => {
      Promise.all([
        this.announce(this.libp2p),
        this.announceModelFn(this.libp2p, this.repoId),
      ]).catch(err => {
        console.error('[BlockRegistry] re-announce failed:', err)
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
