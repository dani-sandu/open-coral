import { describe, it, expect } from 'bun:test'
import { PipelineScheduler, splitMicroBatches, concatenateBatches } from '../../src/inference/pipeline-scheduler'
import type { ChainStepWithCandidates } from '../../src/inference/sequence-manager'

describe('splitMicroBatches', () => {
  it('splits evenly into micro-batches', () => {
    const hidden = new Float32Array(8 * 4)
    const batches = splitMicroBatches(hidden, 8, 4, 4)
    expect(batches).toHaveLength(2)
    expect(batches[0].nTokens).toBe(4)
    expect(batches[0].data.length).toBe(16)
    expect(batches[1].nTokens).toBe(4)
  })

  it('handles remainder batch', () => {
    const hidden = new Float32Array(5 * 2)
    const batches = splitMicroBatches(hidden, 5, 2, 3)
    expect(batches).toHaveLength(2)
    expect(batches[0].nTokens).toBe(3)
    expect(batches[1].nTokens).toBe(2)
  })

  it('single batch when N < M', () => {
    const hidden = new Float32Array(2 * 4)
    const batches = splitMicroBatches(hidden, 2, 4, 128)
    expect(batches).toHaveLength(1)
    expect(batches[0].nTokens).toBe(2)
  })
})

describe('concatenateBatches', () => {
  it('concatenates multiple Float32Arrays', () => {
    const a = new Float32Array([1, 2, 3])
    const b = new Float32Array([4, 5])
    const result = concatenateBatches([a, b])
    expect(Array.from(result)).toEqual([1, 2, 3, 4, 5])
  })

  it('handles single array', () => {
    const a = new Float32Array([1, 2])
    expect(Array.from(concatenateBatches([a]))).toEqual([1, 2])
  })
})

describe('PipelineScheduler', () => {
  it('falls back to sequential for single batch', async () => {
    let forwardCalls = 0

    const chain: ChainStepWithCandidates[] = [
      { candidates: [{ peerId: 'local', blockStart: 0, blockEnd: 3 }], blockStart: 0, blockEnd: 3 },
    ]

    const scheduler = new PipelineScheduler({
      chain,
      nEmbd: 4,
      microBatchSize: 128,
      executeStep: async (_step, input, _nTokens) => {
        forwardCalls++
        return input
      },
    })

    const hidden = new Float32Array(2 * 4).fill(0.5)
    const result = await scheduler.prefill(hidden, 2)
    expect(result.length).toBe(2 * 4)
    expect(forwardCalls).toBe(1)
  })

  it('pipelines multiple batches across multiple steps', async () => {
    const callOrder: string[] = []

    const chain: ChainStepWithCandidates[] = [
      { candidates: [{ peerId: 'A', blockStart: 0, blockEnd: 3 }], blockStart: 0, blockEnd: 3 },
      { candidates: [{ peerId: 'B', blockStart: 4, blockEnd: 7 }], blockStart: 4, blockEnd: 7 },
    ]

    const scheduler = new PipelineScheduler({
      chain,
      nEmbd: 2,
      microBatchSize: 2,
      executeStep: async (step, input, nTokens) => {
        callOrder.push(`s${step.blockStart}-b${nTokens}`)
        return input
      },
    })

    const hidden = new Float32Array(4 * 2).fill(1.0)
    const result = await scheduler.prefill(hidden, 4)
    expect(result.length).toBe(4 * 2)
    expect(callOrder).toHaveLength(4)
  })

  it('preserves KV cache ordering (batch b-1 before batch b at same step)', async () => {
    const order: [number, number][] = []

    const chain: ChainStepWithCandidates[] = [
      { candidates: [{ peerId: 'A', blockStart: 0, blockEnd: 1 }], blockStart: 0, blockEnd: 1 },
      { candidates: [{ peerId: 'B', blockStart: 2, blockEnd: 3 }], blockStart: 2, blockEnd: 3 },
    ]

    const scheduler = new PipelineScheduler({
      chain,
      nEmbd: 2,
      microBatchSize: 2,
      executeStep: async (step, input, _nTokens) => {
        const stepIdx = chain.indexOf(step)
        const batchIdx = order.filter(([s]) => s === stepIdx).length
        order.push([stepIdx, batchIdx])
        await new Promise(r => setTimeout(r, 10))
        return input
      },
    })

    const hidden = new Float32Array(6 * 2).fill(1.0)
    await scheduler.prefill(hidden, 6)

    for (const stepIdx of [0, 1]) {
      const batches = order.filter(([s]) => s === stepIdx).map(([, b]) => b)
      expect(batches).toEqual([0, 1, 2])
    }
  })
})
