import { describe, it, expect } from 'bun:test'
import { runInferenceStream } from '../../src/main/inference-orchestrator'
import type { AsyncBlockRunner } from '../../src/inference/native-worker'
import type { NativeTokenizer } from '../../src/inference/native-tokenizer'

function sharpLogits(vocabSize: number, nTokens: number, id: number): Float32Array {
  const f = new Float32Array(nTokens * vocabSize)
  for (let t = 0; t < nTokens; t++) f[t * vocabSize + id] = 100
  return f
}

function makeMockRunner(vocabSize: number, logitsQueue: Float32Array[]) {
  let queueIdx = 0
  const calls = { openSession: 0, closeSession: 0 }
  const runner = {
    vocabSize,
    hiddenSize: 16,
    openSession: async (_maxSeqLen: number) => { calls.openSession++; return 1 },
    closeSession: async (_id: number) => { calls.closeSession++ },
    sessionDecodeLogitsAll: async (_sid: number, tokens: Int32Array) =>
      logitsQueue[queueIdx++] ?? new Float32Array(tokens.length * vocabSize),
    sessionDecodeLogits: async (_sid: number, tokens: Int32Array) =>
      logitsQueue[queueIdx++] ?? new Float32Array(tokens.length * vocabSize),
    sessionRollback: async () => {},
  }
  return { runner: runner as unknown as AsyncBlockRunner, calls }
}

function makeMockTokenizer(eosTokenId: number): NativeTokenizer {
  return {
    eosTokenId,
    endOfTurnTokenId: undefined,
    decodeToken: async (id: number) => `[${id}]`,
  } as unknown as NativeTokenizer
}

describe('runInferenceStream', () => {
  it('yields decoded pieces for each generated token', async () => {
    const vocabSize = 4
    const eosId = 3
    const { runner } = makeMockRunner(vocabSize, [
      sharpLogits(vocabSize, 1, 1), // prefill → token 1
      sharpLogits(vocabSize, 1, eosId), // next → EOS
    ])
    const tokenizer = makeMockTokenizer(eosId)

    const pieces: string[] = []
    for await (const p of runInferenceStream({
      runner, tokenizer, promptIds: new Int32Array([0]), maxTokens: 10, requestId: 'r1',
    })) {
      pieces.push(p)
    }
    expect(pieces).toEqual(['[1]'])
  })

  it('opens and closes the session around streaming', async () => {
    const vocabSize = 4
    const eosId = 3
    const { runner, calls } = makeMockRunner(vocabSize, [sharpLogits(vocabSize, 1, eosId)])
    const tokenizer = makeMockTokenizer(eosId)

    // consume the full generator
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    for await (const _ of runInferenceStream({
      runner, tokenizer, promptIds: new Int32Array([0]), maxTokens: 5, requestId: 'r1',
    })) { /* drain */ }

    expect(calls.openSession).toBe(1)
    expect(calls.closeSession).toBe(1)
  })

  it('closes the session even when consumer breaks early', async () => {
    const vocabSize = 4
    const eosId = 3
    // many non-EOS tokens so stream would go on forever without break
    const queue = Array.from({ length: 20 }, () => sharpLogits(vocabSize, 1, 1))
    const { runner, calls } = makeMockRunner(vocabSize, queue)
    const tokenizer = makeMockTokenizer(eosId)

    const gen = runInferenceStream({
      runner, tokenizer, promptIds: new Int32Array([0]), maxTokens: 100, requestId: 'r1',
    })
    await gen.next() // get one token
    await gen.return(undefined) // early exit

    expect(calls.closeSession).toBe(1)
  })
})
