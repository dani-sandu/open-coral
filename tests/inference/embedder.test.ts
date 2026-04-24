import { describe, it, expect } from 'bun:test'
import { RunnerEmbedder } from '../../src/inference/runner-embedder'
import type { AsyncBlockRunner } from '../../src/inference/native-worker'

describe('RunnerEmbedder', () => {
  it('nEmbd matches runner.hiddenSize', () => {
    const fake = { hiddenSize: 128, embedTokens: async () => new Float32Array(0) }
    const emb = new RunnerEmbedder(fake as unknown as AsyncBlockRunner)
    expect(emb.nEmbd).toBe(128)
  })

  it('embed returns nTokens × nEmbd floats by delegating to runner', async () => {
    const fake = {
      hiddenSize: 4,
      embedTokens: async (ids: Int32Array) => {
        const out = new Float32Array(ids.length * 4)
        for (let i = 0; i < ids.length; i++) out[i * 4] = ids[i]
        return out
      },
    }
    const emb = new RunnerEmbedder(fake as unknown as AsyncBlockRunner)
    const out = await emb.embed(new Int32Array([7, 8, 9]))
    expect(out.length).toBe(12)
    expect(out[0]).toBe(7)
    expect(out[4]).toBe(8)
    expect(out[8]).toBe(9)
  })
})
