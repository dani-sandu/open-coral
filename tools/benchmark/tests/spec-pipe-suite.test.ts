import { describe, it, expect } from 'bun:test'
import { runSpecPipeSuite } from '../src/suites/spec-pipe'
import type { BenchmarkEvent } from '../src/types'

describe('spec-pipe suite', () => {
  it('emits one specpipe:sample per (depth, acceptance, hops) cell', async () => {
    const events: BenchmarkEvent[] = []
    await runSpecPipeSuite({
      hops: [2, 4],
      acceptanceRates: [0.3, 0.9],
      pipelineDepths: [1, 2],
      runsPerCell: 3,
      simHopLatencyMs: 5,
      sink: e => events.push(e),
    })
    const samples = events.filter(e => e.type === 'specpipe:sample')
    // 2 hops × 2 acceptances × 2 depths = 8 cells
    expect(samples.length).toBe(8)
    for (const s of samples) {
      if (s.type !== 'specpipe:sample') continue
      expect(s.msPerToken).toBeGreaterThan(0)
      expect(s.rollbacks).toBeGreaterThanOrEqual(0)
    }
  })

  // Extended timeout: 4 hops × 20 ms × ~8 spec iterations × 5 runs = ~3.2 s for
  // depth=1 alone; on Windows the busy-wait sleep has ms-level granularity so
  // the cumulative cell time gets close to bun's default 5 s. 15 s is generous.
  it('depth=2 has lower msPerToken than depth=1 at high acceptance and M>=4', async () => {
    const events: BenchmarkEvent[] = []
    await runSpecPipeSuite({
      hops: [4],
      acceptanceRates: [0.9],
      pipelineDepths: [1, 2],
      runsPerCell: 5,
      simHopLatencyMs: 20,
      sink: e => events.push(e),
    })
    const byKey = (depth: 1 | 2) => events.find(e => e.type === 'specpipe:sample' && e.pipelineDepth === depth) as Extract<BenchmarkEvent, { type: 'specpipe:sample' }>
    const d1 = byKey(1)
    const d2 = byKey(2)
    expect(d1.msPerToken).toBeGreaterThan(d2.msPerToken)
  }, 15000)
})
