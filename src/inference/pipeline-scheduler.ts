import type { ChainStepWithCandidates } from './sequence-manager'

export interface MicroBatch {
  data: Float32Array
  nTokens: number
}

export function splitMicroBatches(
  hidden: Float32Array,
  N: number,
  nEmbd: number,
  M: number,
): MicroBatch[] {
  const batches: MicroBatch[] = []
  for (let offset = 0; offset < N; offset += M) {
    const end = Math.min(offset + M, N)
    const nTokens = end - offset
    batches.push({
      data: hidden.subarray(offset * nEmbd, end * nEmbd),
      nTokens,
    })
  }
  return batches
}

export function concatenateBatches(results: Float32Array[]): Float32Array {
  const totalFloats = results.reduce((sum, r) => sum + r.length, 0)
  const output = new Float32Array(totalFloats)
  let offset = 0
  for (const r of results) {
    output.set(r, offset)
    offset += r.length
  }
  return output
}

export type StepExecutor = (
  step: ChainStepWithCandidates,
  input: Float32Array,
  nTokens: number,
) => Promise<Float32Array>

export interface PipelineSchedulerOptions {
  chain: ChainStepWithCandidates[]
  nEmbd: number
  microBatchSize: number
  executeStep: StepExecutor
}

export class PipelineScheduler {
  private readonly chain: ChainStepWithCandidates[]
  private readonly nEmbd: number
  private readonly microBatchSize: number
  private readonly executeStep: StepExecutor

  constructor(opts: PipelineSchedulerOptions) {
    this.chain = opts.chain
    this.nEmbd = opts.nEmbd
    this.microBatchSize = opts.microBatchSize
    this.executeStep = opts.executeStep
  }

  async prefill(hidden: Float32Array, totalTokens: number): Promise<Float32Array> {
    const batches = splitMicroBatches(hidden, totalTokens, this.nEmbd, this.microBatchSize)
    const K = batches.length
    const S = this.chain.length

    if (K === 1) {
      let current = batches[0].data
      for (const step of this.chain) {
        current = await this.executeStep(step, current, batches[0].nTokens)
      }
      return current
    }

    const grid: Promise<Float32Array>[][] = []
    for (let b = 0; b < K; b++) {
      grid[b] = []
      for (let s = 0; s < S; s++) {
        const dataDep: Promise<Float32Array> = s > 0
          ? grid[b][s - 1]
          : Promise.resolve(batches[b].data)
        const cacheDep: Promise<unknown> = b > 0
          ? grid[b - 1][s]
          : Promise.resolve(null)

        const batchNTokens = batches[b].nTokens
        const step = this.chain[s]

        grid[b][s] = Promise.all([dataDep, cacheDep]).then(([input]) => {
          return this.executeStep(step, input, batchNTokens)
        })
      }
    }

    const results = await Promise.all(grid.map(row => row[S - 1]))
    return concatenateBatches(results)
  }
}
