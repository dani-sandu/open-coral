import { describe, it, expect } from 'bun:test'
import { runInference } from '../../src/main/inference-orchestrator'
import type { AsyncBlockRunner } from '../../src/inference/native-worker'
import type { NativeTokenizer } from '../../src/inference/native-tokenizer'

function sharpLogits(vocabSize: number, nTokens: number, id: number): Float32Array {
  const f = new Float32Array(nTokens * vocabSize)
  for (let t = 0; t < nTokens; t++) f[t * vocabSize + id] = 100
  return f
}

function makeMockRunner(vocabSize: number, logitsQueue: Float32Array[]) {
  const calls = { openSession: 0, closeSession: 0, decodeLogitsAll: 0, decodeLogits: 0 }
  let queueIdx = 0
  const runner = {
    vocabSize,
    hiddenSize: 16,
    openSession: async (_maxSeqLen: number) => { calls.openSession++; return 1 },
    closeSession: async (_id: number) => { calls.closeSession++ },
    sessionDecodeLogitsAll: async (_sid: number, tokens: Int32Array) => {
      calls.decodeLogitsAll++
      return logitsQueue[queueIdx++] ?? new Float32Array(tokens.length * vocabSize)
    },
    sessionDecodeLogits: async (_sid: number, tokens: Int32Array) => {
      calls.decodeLogits++
      return logitsQueue[queueIdx++] ?? new Float32Array(tokens.length * vocabSize)
    },
    sessionRollback: async () => {},
  }
  return { runner: runner as unknown as AsyncBlockRunner, calls }
}

function makeMockTokenizer(eosTokenId: number, promptIds: number[], decoded: string): NativeTokenizer {
  return {
    eosTokenId,
    endOfTurnTokenId: undefined,
    encodeChat: async (_text: string) => new Int32Array(promptIds),
    decode: async (_ids: number[]) => decoded,
    decodeToken: async (_id: number) => '?',
  } as unknown as NativeTokenizer
}

describe('runInference', () => {
  it('opens and closes a session around generation', async () => {
    const vocabSize = 4
    const eos = 3
    const { runner, calls } = makeMockRunner(vocabSize, [sharpLogits(vocabSize, 1, eos)])
    const tokenizer = makeMockTokenizer(eos, [1], '')
    await runInference({ runner, tokenizer, prompt: 'hi', maxTokens: 5, requestId: 'r1' })
    expect(calls.openSession).toBe(1)
    expect(calls.closeSession).toBe(1)
  })

  it('returns promptLength and genResult from speculative session', async () => {
    const vocabSize = 4
    const eos = 3
    // prompt [1,2] → prefill logits 2 rows, last row sharp on EOS → no tokens generated
    const { runner } = makeMockRunner(vocabSize, [sharpLogits(vocabSize, 2, eos)])
    const tokenizer = makeMockTokenizer(eos, [1, 2], '')
    const out = await runInference({ runner, tokenizer, prompt: 'hi', maxTokens: 5, requestId: 'r1' })
    expect(out.promptLength).toBe(2)
    expect(out.genResult.tokenIds).toEqual([])
    expect(typeof out.inferenceDurationMs).toBe('number')
  })

  it('closes the session even when generation throws', async () => {
    const vocabSize = 4
    const { runner, calls } = makeMockRunner(vocabSize, [])
    // Make the decode throw immediately (empty queue + throwing runner)
    ;(runner as unknown as { sessionDecodeLogitsAll: (sid: number, ids: Int32Array) => Promise<Float32Array> })
      .sessionDecodeLogitsAll = async () => { throw new Error('decode exploded') }
    const tokenizer = makeMockTokenizer(3, [1, 2], '')

    let caught: unknown = null
    try {
      await runInference({ runner, tokenizer, prompt: 'hi', maxTokens: 5, requestId: 'r1' })
    } catch (err) {
      caught = err
    }
    expect((caught as Error).message).toBe('decode exploded')
    expect(calls.openSession).toBe(1)
    expect(calls.closeSession).toBe(1)
  })
})
