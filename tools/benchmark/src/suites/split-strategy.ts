import { SequenceManager } from '../../../../src/inference/sequence-manager'
import type { NodeIdentity } from '../../../../src/main/identity'
import type { VirtualNode } from '../sim/virtual-node'
import { generatePartitions } from './partitions'
import { buildChain } from './latency'
import type { EventSink } from '../types'

export interface SplitStrategySuiteOptions {
  client: VirtualNode
  workers: VirtualNode[]
  identity: NodeIdentity
  modelBlocks: number
  hiddenSize: number
  nodeCounts: number[]
  runsPerCell: number
  sink: EventSink
}

function median(values: number[]): number {
  if (values.length === 0) return -1
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
}

export async function runSplitStrategySuite(opts: SplitStrategySuiteOptions): Promise<void> {
  const { client, workers, identity, modelBlocks, hiddenSize, nodeCounts, runsPerCell, sink } = opts
  sink({ type: 'suite:start', suite: 'split-strategy', mode: 'sim' })

  for (const nodeCount of nodeCounts) {
    for (const partition of generatePartitions(modelBlocks, nodeCount)) {
      try {
        if (workers.length < nodeCount) {
          throw new Error(`need ${nodeCount} workers, have ${workers.length}`)
        }
        const chainWorkers = workers.slice(0, nodeCount)
        const chain = buildChain(partition.boundaries, chainWorkers)
        const mgr = new SequenceManager({
          node: client, localRunner: null, totalBlocks: modelBlocks, hiddenSize, identity,
        })
        const samples: number[] = []
        for (let r = 0; r < runsPerCell; r++) {
          const input = new Float32Array(1 * hiddenSize).fill(0.1)
          const t0 = Date.now()
          await mgr.runChain(chain, input, 1, `split-${nodeCount}-${partition.strategy}-${r}`)
          samples.push(Date.now() - t0)
        }
        sink({
          type: 'heatmap:cell',
          nodeCount,
          partitionBoundaries: partition.boundaries,
          latencyMs: median(samples),
          mode: 'sim',
        })
      } catch (err) {
        sink({
          type: 'error',
          suite: 'split-strategy',
          message: `nodeCount=${nodeCount} strategy=${partition.strategy}: ${err instanceof Error ? err.message : String(err)}`,
          mode: 'sim',
        })
      }
    }
  }

  sink({ type: 'suite:end', suite: 'split-strategy', mode: 'sim' })
}
