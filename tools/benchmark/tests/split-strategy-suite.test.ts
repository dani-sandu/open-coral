import { describe, it, expect } from 'bun:test'
import { mkdtempSync, rmSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { SimNetwork } from '../src/sim/sim-network'
import { loadOrCreateIdentity } from '../../../src/main/identity'
import { registerInferenceHandlerV3 } from '../../../src/p2p/inference-protocol'
import { runSplitStrategySuite } from '../src/suites/split-strategy'
import type { BenchmarkEvent } from '../src/types'

describe('runSplitStrategySuite', () => {
  it('emits a heatmap:cell for every (nodeCount, partition) combination', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'coral-split-suite-'))
    const identity = await loadOrCreateIdentity(dir)
    const nEmbd = 8

    const net = new SimNetwork({ latencyMeanMs: 1, latencyJitterMs: 0, modelBlocks: 8 })
    const client = await net.addNode()
    // Pre-create enough workers for the largest nodeCount in the sweep.
    const workers = []
    for (let i = 0; i < 4; i++) {
      const w = await net.addNode()
      await registerInferenceHandlerV3(w.libp2p, async (input) => input, identity)
      workers.push(w)
    }

    const events: BenchmarkEvent[] = []
    await runSplitStrategySuite({
      client, workers, identity, modelBlocks: 8, hiddenSize: nEmbd,
      nodeCounts: [2, 4], runsPerCell: 3, sink: e => events.push(e),
    })

    const cells = events.filter(e => e.type === 'heatmap:cell')
    // 2 node counts × 3 strategies (equal/front-heavy/back-heavy) = 6 cells.
    expect(cells.length).toBe(6)
    for (const c of cells) {
      if (c.type !== 'heatmap:cell') continue
      expect(c.partitionBoundaries[0]).toBe(0)
      expect(c.partitionBoundaries[c.partitionBoundaries.length - 1]).toBe(8)
      expect(c.latencyMs).toBeGreaterThanOrEqual(0)
    }
    rmSync(dir, { recursive: true })
  })
})
