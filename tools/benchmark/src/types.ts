export type BenchmarkMode = 'sim' | 'real'

export type BenchmarkEvent =
  | { type: 'suite:start'; suite: string; mode: BenchmarkMode }
  | { type: 'suite:end'; suite: string; mode: BenchmarkMode }
  | { type: 'latency:sample'; runIndex: number; stepRtts: Record<string, number>; totalMs: number; mode: BenchmarkMode }
  | { type: 'throughput:sample'; bytesPerSec: number; tokensPerSec: number; batchSize: number; mode: BenchmarkMode }
  | { type: 'heatmap:cell'; nodeCount: number; partitionBoundaries: number[]; latencyMs: number; mode: BenchmarkMode }
  | { type: 'error'; suite: string; message: string; mode: BenchmarkMode }
  | { type: 'done' }

export function isTerminalEvent(e: BenchmarkEvent): boolean {
  return e.type === 'done'
}

/** A named contiguous partition of the model's blocks across N nodes. */
export interface Partition {
  /** e.g. 'equal' | 'front-heavy' | 'back-heavy' */
  strategy: string
  /** sorted block-index boundaries, e.g. [0, 8, 24, 32] → ranges [0–7],[8–23],[24–31] */
  boundaries: number[]
}

export interface SimConfig {
  modelBlocks: number
  latencyMeanMs: number
  latencyJitterMs: number
}

/** Aggregated results collected across a run, written to the baseline artifact. */
export interface SuiteResult {
  events: BenchmarkEvent[]
  startedAt: string
  finishedAt: string
  config: SimConfig
}

export type EventSink = (e: BenchmarkEvent) => void
