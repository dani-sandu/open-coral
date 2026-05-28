import { describe, it, expect } from 'bun:test'
import { mkdtempSync, rmSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { SimNetwork } from '../src/sim/sim-network'
import { loadOrCreateIdentity } from '../../../src/main/identity'
import { registerInferenceHandlerV3 } from '../../../src/p2p/inference-protocol'
import { runLatencySuite, buildChain } from '../src/suites/latency'
import type { BenchmarkEvent } from '../src/types'

describe('buildChain', () => {
  it('maps boundaries + nodes to inclusive ChainSteps', async () => {
    const net = new SimNetwork({ latencyMeanMs: 1, latencyJitterMs: 0, modelBlocks: 4 })
    const w1 = await net.addNode()
    const w2 = await net.addNode()
    const chain = buildChain([0, 2, 4], [w1, w2])
    expect(chain.length).toBe(2)
    expect(chain[0]).toMatchObject({ peerId: w1.peerId, blockStart: 0, blockEnd: 1 })
    expect(chain[1]).toMatchObject({ peerId: w2.peerId, blockStart: 2, blockEnd: 3 })
  })
})

describe('runLatencySuite', () => {
  it('emits suite:start, one latency:sample per run with stepRtts, and suite:end', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'coral-lat-suite-'))
    const identity = await loadOrCreateIdentity(dir)
    const nEmbd = 8

    const net = new SimNetwork({ latencyMeanMs: 3, latencyJitterMs: 1, modelBlocks: 2 })
    const client = await net.addNode()
    const w1 = await net.addNode()
    const w2 = await net.addNode()
    for (const w of [w1, w2]) {
      await registerInferenceHandlerV3(w.libp2p, async (input) => input, identity)
    }

    const events: BenchmarkEvent[] = []
    await runLatencySuite({
      client, workers: [w1, w2], identity,
      boundaries: [0, 1, 2], modelBlocks: 2, hiddenSize: nEmbd,
      runs: 5, sink: e => events.push(e),
    })

    expect(events[0]).toMatchObject({ type: 'suite:start', suite: 'latency' })
    expect(events[events.length - 1]).toMatchObject({ type: 'suite:end', suite: 'latency' })
    const samples = events.filter(e => e.type === 'latency:sample')
    expect(samples.length).toBe(5)
    for (const s of samples) {
      if (s.type !== 'latency:sample') continue
      expect(s.totalMs).toBeGreaterThan(0)
      expect(Object.keys(s.stepRtts).length).toBe(2) // one per worker
    }
    rmSync(dir, { recursive: true })
  })
})
