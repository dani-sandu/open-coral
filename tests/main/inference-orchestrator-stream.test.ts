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
    decode: async (ids: number[]) => ids.map(id => `[${id}]`).join(''),
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

describe('runInferenceStream — IPC batching', () => {
  it('coalesces decodeToken calls into batches via tokenizer.decode', async () => {
    const vocabSize = 4
    const eosId = 3
    // Always-sharp-on-token-1 logits; we cap via maxTokens so the exact emission
    // count is deterministic regardless of how speculative decoding batches.
    // Logits are padded to 16 positions so spec batch reads of >1 position
    // still hit a sharp peak rather than the zero-padded tail.
    const queue: Float32Array[] = []
    for (let i = 0; i < 30; i++) {
      queue.push(sharpLogits(vocabSize, 16, 1))
    }
    const { runner } = makeMockRunner(vocabSize, queue)

    let decodeTokenCalls = 0
    let decodeBatchCalls = 0
    const tokenizer = {
      eosTokenId: eosId,
      endOfTurnTokenId: undefined,
      decodeToken: async (id: number) => { decodeTokenCalls++; return `[${id}]` },
      decode: async (ids: number[]) => {
        decodeBatchCalls++
        return ids.map(id => `[${id}]`).join('')
      },
    } as unknown as import('../../src/inference/native-tokenizer').NativeTokenizer

    const pieces: string[] = []
    for await (const p of runInferenceStream({
      runner, tokenizer, promptIds: new Int32Array([0]), maxTokens: 8, requestId: 'r1',
    })) {
      pieces.push(p)
    }

    // With batching, we expect decode() to be used (not decodeToken per token).
    expect(decodeBatchCalls).toBeGreaterThan(0)
    expect(decodeTokenCalls).toBe(0)

    // The concatenated output must still equal one piece per generated token.
    expect(pieces.join('')).toBe('[1][1][1][1][1][1][1][1]')
    // And we should see fewer yields than tokens (batching reduces yield count).
    expect(pieces.length).toBeLessThan(8)
  })

  it('flushes the trailing partial batch on stream end', async () => {
    const vocabSize = 4
    const eosId = 3
    // 3 tokens — does not hit batch size of 4, so a tail flush is needed.
    // Logits padded to 16 positions so spec batch reads stay deterministic;
    // maxTokens caps the total emission so the count is independent of spec batching.
    const queue = Array.from({ length: 10 }, () => sharpLogits(vocabSize, 16, 1))
    const { runner } = makeMockRunner(vocabSize, queue)

    const tokenizer = {
      eosTokenId: eosId,
      endOfTurnTokenId: undefined,
      decodeToken: async (id: number) => `[${id}]`,
      decode: async (ids: number[]) => ids.map(id => `[${id}]`).join(''),
    } as unknown as import('../../src/inference/native-tokenizer').NativeTokenizer

    const pieces: string[] = []
    for await (const p of runInferenceStream({
      runner, tokenizer, promptIds: new Int32Array([0]), maxTokens: 3, requestId: 'r1',
    })) {
      pieces.push(p)
    }

    expect(pieces.join('')).toBe('[1][1][1]')
    expect(pieces.length).toBe(1)  // single tail-flush yield, not 3 per-token yields
  })

  it('emits nothing and calls neither decode nor decodeToken when first token is EOS', async () => {
    const vocabSize = 4
    const eosId = 3
    // Prefill samples the EOS token immediately — no tokens are ever generated.
    const { runner } = makeMockRunner(vocabSize, [sharpLogits(vocabSize, 1, eosId)])

    let decodeCalls = 0
    let decodeTokenCalls = 0
    const tokenizer = {
      eosTokenId: eosId,
      endOfTurnTokenId: undefined,
      decodeToken: async () => { decodeTokenCalls++; return '' },
      decode: async () => { decodeCalls++; return '' },
    } as unknown as import('../../src/inference/native-tokenizer').NativeTokenizer

    const pieces: string[] = []
    for await (const p of runInferenceStream({
      runner, tokenizer, promptIds: new Int32Array([0]), maxTokens: 10, requestId: 'r1',
    })) {
      pieces.push(p)
    }

    expect(pieces).toEqual([])
    expect(decodeCalls).toBe(0)
    expect(decodeTokenCalls).toBe(0)
  })
})
