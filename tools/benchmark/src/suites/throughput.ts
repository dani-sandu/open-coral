import { SequenceManager } from '../../../../src/inference/sequence-manager'
import type { NodeIdentity } from '../../../../src/main/identity'
import type { VirtualNode } from '../sim/virtual-node'
import { buildChain } from './latency'
import type { EventSink } from '../types'

export interface ThroughputSuiteOptions {
  client: VirtualNode
  workers: VirtualNode[]
  identity: NodeIdentity
  boundaries: number[]
  modelBlocks: number
  hiddenSize: number
  batchSizes: number[]
  sink: EventSink
}

export async function runThroughputSuite(opts: ThroughputSuiteOptions): Promise<void> {
  const { client, workers, identity, boundaries, modelBlocks, hiddenSize, batchSizes, sink } = opts
  sink({ type: 'suite:start', suite: 'throughput', mode: 'sim' })

  const chain = buildChain(boundaries, workers)
  const mgr = new SequenceManager({
    node: client, localRunner: null, totalBlocks: modelBlocks, hiddenSize, identity,
  })

  for (const batchSize of batchSizes) {
    const input = new Float32Array(batchSize * hiddenSize).fill(0.1)
    const hopLog: import('../../../../src/inference/sequence-manager').HopTiming[] = []
    const t0 = Date.now()
    await mgr.runChain(chain, input, batchSize, `tp-${batchSize}`, hopLog)
    const seconds = Math.max((Date.now() - t0) / 1000, 1e-6)

    // Sum actual encoded request bytes per hop (fp16 vs fp32 shows here).
    // Local hops report no wireBytes — fall back to the fp32 tensor size.
    let totalBytes = 0
    for (const hop of hopLog) {
      totalBytes += hop.wireBytes ?? (batchSize * hiddenSize * 4)
    }
    sink({
      type: 'throughput:sample',
      bytesPerSec: totalBytes / seconds,
      tokensPerSec: batchSize / seconds,
      batchSize,
      totalWireBytes: totalBytes,
      mode: 'sim',
    })
  }

  sink({ type: 'suite:end', suite: 'throughput', mode: 'sim' })
}
