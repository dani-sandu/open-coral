import { describe, it, expect } from 'bun:test'
import { mkdtempSync, rmSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { SimNetwork } from '../src/sim/sim-network'
import { loadOrCreateIdentity } from '../../../src/main/identity'
import { registerInferenceHandlerV3 } from '../../../src/p2p/inference-protocol'
import { runThroughputSuite } from '../src/suites/throughput'
import type { BenchmarkEvent } from '../src/types'

describe('runThroughputSuite', () => {
  it('emits one throughput:sample per batch size with positive bytes/sec and tokens/sec', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'coral-tp-suite-'))
    const identity = await loadOrCreateIdentity(dir)
    const nEmbd = 8

    const net = new SimNetwork({ latencyMeanMs: 1, latencyJitterMs: 0, modelBlocks: 1 })
    const client = await net.addNode()
    const worker = await net.addNode()
    await registerInferenceHandlerV3(worker.libp2p, async (input) => input, identity)

    const events: BenchmarkEvent[] = []
    await runThroughputSuite({
      client, workers: [worker], identity,
      boundaries: [0, 1], modelBlocks: 1, hiddenSize: nEmbd,
      batchSizes: [1, 4, 16], sink: e => events.push(e),
    })

    const samples = events.filter(e => e.type === 'throughput:sample')
    expect(samples.length).toBe(3)
    for (const s of samples) {
      if (s.type !== 'throughput:sample') continue
      expect(s.bytesPerSec).toBeGreaterThan(0)
      expect(s.tokensPerSec).toBeGreaterThan(0)
    }
    rmSync(dir, { recursive: true })
  })
})
