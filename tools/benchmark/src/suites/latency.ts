import { SequenceManager, type ChainStep } from '../../../../src/inference/sequence-manager'
import { PeerLatencyTracker } from '../../../../src/p2p/peer-latency'
import type { NodeIdentity } from '../../../../src/main/identity'
import type { VirtualNode } from '../sim/virtual-node'
import { boundariesToRanges } from './partitions'
import type { EventSink } from '../types'

/** Map partition boundaries + an ordered worker list to a ChainStep[]. */
export function buildChain(boundaries: number[], workers: VirtualNode[]): ChainStep[] {
  const ranges = boundariesToRanges(boundaries)
  if (ranges.length > workers.length) {
    throw new Error(`buildChain: ${ranges.length} ranges but only ${workers.length} workers`)
  }
  return ranges.map((r, i) => ({
    peerId: workers[i].peerId,
    blockStart: r.blockStart,
    blockEnd: r.blockEnd,
    multiaddr: workers[i].multiaddrs[0],
  }))
}

export interface LatencySuiteOptions {
  client: VirtualNode
  workers: VirtualNode[]
  identity: NodeIdentity
  boundaries: number[]
  modelBlocks: number
  hiddenSize: number
  runs: number
  sink: EventSink
}

export async function runLatencySuite(opts: LatencySuiteOptions): Promise<void> {
  const { client, workers, identity, boundaries, modelBlocks, hiddenSize, runs, sink } = opts
  sink({ type: 'suite:start', suite: 'latency', mode: 'sim' })

  const chain = buildChain(boundaries, workers)
  const usedWorkers = workers.slice(0, chain.length)

  for (let runIndex = 0; runIndex < runs; runIndex++) {
    // Fresh tracker per run so stepRtts reflect this run, not a blended EWMA.
    const tracker = new PeerLatencyTracker()
    const mgr = new SequenceManager({
      node: client, localRunner: null,
      totalBlocks: modelBlocks, hiddenSize, identity, latencyTracker: tracker,
    })
    const input = new Float32Array(1 * hiddenSize).fill(0.1) // 1-token decode step
    const t0 = Date.now()
    await mgr.runChain(chain, input, 1, `lat-${runIndex}`)
    const totalMs = Date.now() - t0

    const stepRtts: Record<string, number> = {}
    for (const w of usedWorkers) stepRtts[w.peerId] = tracker.getEstimate(w.peerId)

    sink({ type: 'latency:sample', runIndex, stepRtts, totalMs, mode: 'sim' })
  }

  sink({ type: 'suite:end', suite: 'latency', mode: 'sim' })
}
