import { describe, it, expect } from 'bun:test'
import type { BenchmarkEvent } from '../src/types'
import { isTerminalEvent } from '../src/types'

describe('BenchmarkEvent', () => {
  it('isTerminalEvent is true only for the done event', () => {
    const done: BenchmarkEvent = { type: 'done' }
    const sample: BenchmarkEvent = {
      type: 'latency:sample', runIndex: 0, stepRtts: { p1: 10 }, totalMs: 12, mode: 'sim',
    }
    expect(isTerminalEvent(done)).toBe(true)
    expect(isTerminalEvent(sample)).toBe(false)
  })
})
