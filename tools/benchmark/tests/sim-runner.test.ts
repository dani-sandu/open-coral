import { describe, it, expect } from 'bun:test'
import { runSimBenchmark, parseArgs } from '../src/sim-runner'
import type { BenchmarkEvent } from '../src/types'

describe('parseArgs', () => {
  it('applies defaults and overrides', () => {
    const def = parseArgs([])
    expect(def.mode).toBe('sim')
    expect(def.modelBlocks).toBe(32)
    expect(def.runs).toBe(50)
    const over = parseArgs(['--model-blocks', '16', '--runs', '5', '--nodes', '2,4', '--latency-mean', '10', '--latency-jitter', '3'])
    expect(over.modelBlocks).toBe(16)
    expect(over.runs).toBe(5)
    expect(over.nodes).toEqual([2, 4])
    expect(over.latencyMeanMs).toBe(10)
    expect(over.latencyJitterMs).toBe(3)
  })
})

describe('runSimBenchmark', () => {
  it('runs all four suites and ends with a done event', async () => {
    const events: BenchmarkEvent[] = []
    await runSimBenchmark(
      { mode: 'sim', modelBlocks: 8, nodes: [2, 4], runs: 3, latencyMeanMs: 1, latencyJitterMs: 0, port: 0 },
      e => events.push(e),
    )
    const suiteStarts = events.filter(e => e.type === 'suite:start').map(e => (e as any).suite).sort()
    expect(suiteStarts).toEqual(['latency', 'spec-pipe', 'split-strategy', 'throughput'])
    expect(events[events.length - 1]).toMatchObject({ type: 'done' })
  }, 30000)
})
