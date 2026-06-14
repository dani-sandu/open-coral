import { mkdtempSync, rmSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { SimNetwork } from './sim/sim-network'
import { loadOrCreateIdentity } from '../../../src/main/identity'
import { registerInferenceHandlerV3, registerInferenceHandlerV4 } from '../../../src/p2p/inference-protocol'
import { generatePartitions } from './suites/partitions'
import { runLatencySuite } from './suites/latency'
import { runThroughputSuite } from './suites/throughput'
import { runSplitStrategySuite } from './suites/split-strategy'
import { runSpecPipeSuite } from './suites/spec-pipe'
import type { BenchmarkEvent, EventSink } from './types'

export interface CliConfig {
  mode: 'sim' | 'real'
  modelBlocks: number
  nodes: number[]
  runs: number
  latencyMeanMs: number
  latencyJitterMs: number
  port: number
}

export function parseArgs(argv: string[]): CliConfig {
  const get = (flag: string): string | undefined => {
    const i = argv.indexOf(flag)
    return i >= 0 && i + 1 < argv.length ? argv[i + 1] : undefined
  }
  return {
    mode: (get('--mode') as 'sim' | 'real') ?? 'sim',
    modelBlocks: Number(get('--model-blocks') ?? 32),
    nodes: (get('--nodes') ?? '2,4,8,16').split(',').map(n => Number(n.trim())).filter(n => n > 0),
    runs: Number(get('--runs') ?? 50),
    latencyMeanMs: Number(get('--latency-mean') ?? 20),
    latencyJitterMs: Number(get('--latency-jitter') ?? 8),
    port: Number(get('--port') ?? 4321),
  }
}

const THROUGHPUT_BATCHES = [1, 4, 16, 64, 256]

export async function runSimBenchmark(cfg: CliConfig, sink: EventSink): Promise<void> {
  const dir = mkdtempSync(join(tmpdir(), 'coral-bench-'))
  try {
    const identity = await loadOrCreateIdentity(dir)
    const hiddenSize = 4096 // representative of a real model's hidden dim
    const maxNodes = Math.max(...cfg.nodes, 2)

    const net = new SimNetwork({
      modelBlocks: cfg.modelBlocks,
      latencyMeanMs: cfg.latencyMeanMs,
      latencyJitterMs: cfg.latencyJitterMs,
    })
    const client = await net.addNode()
    const workers = []
    for (let i = 0; i < maxNodes; i++) {
      const w = await net.addNode()
      await registerInferenceHandlerV3(w.libp2p, async (input) => input, identity)
      await registerInferenceHandlerV4(w.libp2p, async (input) => input, identity)
      workers.push(w)
    }

    // Latency + throughput run on the equal partition for the smallest node count.
    const baseNodeCount = Math.min(...cfg.nodes)
    const equal = generatePartitions(cfg.modelBlocks, baseNodeCount).find(p => p.strategy === 'equal')!

    await runLatencySuite({
      client, workers, identity,
      boundaries: equal.boundaries, modelBlocks: cfg.modelBlocks, hiddenSize,
      runs: cfg.runs, sink,
    })
    await runThroughputSuite({
      client, workers, identity,
      boundaries: equal.boundaries, modelBlocks: cfg.modelBlocks, hiddenSize,
      batchSizes: THROUGHPUT_BATCHES, sink,
    })
    await runSplitStrategySuite({
      client, workers, identity,
      modelBlocks: cfg.modelBlocks, hiddenSize,
      nodeCounts: cfg.nodes, runsPerCell: 10, sink,
    })
    await runSpecPipeSuite({
      hops: cfg.nodes,
      acceptanceRates: [0.3, 0.6, 0.9],
      pipelineDepths: [1, 2],
      runsPerCell: cfg.runs >= 30 ? 10 : 3,
      simHopLatencyMs: cfg.latencyMeanMs,
      sink,
    })

    sink({ type: 'done' })
  } finally {
    rmSync(dir, { recursive: true })
  }
}

export type { BenchmarkEvent }
