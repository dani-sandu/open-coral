import { describe, it, expect, beforeAll, afterAll } from 'bun:test'
import { ApiServer, type ApiServerDeps } from '../../src/main/api-server'
import type { AsyncBlockRunner } from '../../src/inference/native-worker'
import type { NativeTokenizer } from '../../src/inference/native-tokenizer'
import type { ModelInfo } from '../../src/main/model-manager'

const TEST_PORT = 39399
const TEST_KEY = 'sk-local-test-key'

function sharpLogits(vocabSize: number, nTokens: number, id: number): Float32Array {
  const f = new Float32Array(nTokens * vocabSize)
  for (let t = 0; t < nTokens; t++) f[t * vocabSize + id] = 100
  return f
}

function makeDeps(eosId = 3, modelLoaded = true): ApiServerDeps {
  const vocabSize = 4
  let queueIdx = 0
  const queue = [
    sharpLogits(vocabSize, 1, 1), // prefill → token 1
    sharpLogits(vocabSize, 1, eosId), // next → EOS
    // extra entries for second call
    sharpLogits(vocabSize, 1, 1),
    sharpLogits(vocabSize, 1, eosId),
  ]
  const runner = {
    vocabSize,
    hiddenSize: 16,
    openSession: async () => 1,
    closeSession: async () => {},
    sessionDecodeLogitsAll: async (_sid: number, tokens: Int32Array) =>
      queue[queueIdx++] ?? new Float32Array(tokens.length * vocabSize),
    sessionDecodeLogits: async (_sid: number, tokens: Int32Array) =>
      queue[queueIdx++] ?? new Float32Array(tokens.length * vocabSize),
    sessionRollback: async () => {},
  } as unknown as AsyncBlockRunner

  const tokenizer = {
    eosTokenId: eosId,
    endOfTurnTokenId: undefined,
    decodeToken: async (id: number) => `T${id}`,
    encodeChatMulti: async () => new Int32Array([0]),
  } as unknown as NativeTokenizer

  const model: ModelInfo = {
    path: '/tmp/test.gguf', architecture: 'llama', totalBlocks: 32,
    hiddenSize: 4096, headCount: 32, fileSizeBytes: 1000, blockTensorSuffixes: [],
    repoId: 'test/model',
  }

  return {
    getRunner: () => (modelLoaded ? runner : null),
    getTokenizer: () => (modelLoaded ? tokenizer : null),
    getModel: () => (modelLoaded ? model : null),
  }
}

let server: ApiServer

beforeAll(async () => {
  server = new ApiServer({ enabled: true, port: TEST_PORT, apiKey: TEST_KEY }, makeDeps())
  await server.start()
})

afterAll(async () => {
  await server.stop()
})

const base = `http://localhost:${TEST_PORT}`
const headers = { 'Authorization': `Bearer ${TEST_KEY}`, 'Content-Type': 'application/json' }

describe('auth', () => {
  it('rejects missing auth header with 401', async () => {
    const res = await fetch(`${base}/health`)
    expect(res.status).toBe(401)
  })

  it('rejects wrong key with 401', async () => {
    const res = await fetch(`${base}/health`, { headers: { 'Authorization': 'Bearer wrong' } })
    expect(res.status).toBe(401)
  })
})

describe('GET /health', () => {
  it('returns status ok and model name', async () => {
    const res = await fetch(`${base}/health`, { headers })
    expect(res.status).toBe(200)
    const body = await res.json() as { status: string; model: string }
    expect(body.status).toBe('ok')
    expect(body.model).toBe('test/model')
  })
})

describe('GET /v1/models', () => {
  it('returns the loaded model', async () => {
    const res = await fetch(`${base}/v1/models`, { headers })
    expect(res.status).toBe(200)
    const body = await res.json() as { data: Array<{ id: string }> }
    expect(body.data[0].id).toBe('test/model')
  })
})

describe('POST /v1/chat/completions', () => {
  it('returns a non-streaming completion', async () => {
    const res = await fetch(`${base}/v1/chat/completions`, {
      method: 'POST', headers,
      body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }], max_tokens: 5, stream: false }),
    })
    expect(res.status).toBe(200)
    const body = await res.json() as { choices: Array<{ message: { content: string } }> }
    expect(typeof body.choices[0].message.content).toBe('string')
  })

  it('streams SSE events with stream:true', async () => {
    const res = await fetch(`${base}/v1/chat/completions`, {
      method: 'POST', headers,
      body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }], max_tokens: 5, stream: true }),
    })
    expect(res.status).toBe(200)
    expect(res.headers.get('content-type')).toContain('text/event-stream')
    const text = await res.text()
    expect(text).toContain('data:')
    expect(text).toContain('[DONE]')
  })

  it('returns 503 when no model loaded', async () => {
    const noModelServer = new ApiServer(
      { enabled: true, port: TEST_PORT + 1, apiKey: TEST_KEY },
      makeDeps(3, false),
    )
    await noModelServer.start()
    try {
      const res = await fetch(`http://localhost:${TEST_PORT + 1}/v1/chat/completions`, {
        method: 'POST', headers,
        body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }], stream: false }),
      })
      expect(res.status).toBe(503)
    } finally {
      await noModelServer.stop()
    }
  })
})

describe('POST /v1/messages', () => {
  it('returns a non-streaming anthropic message', async () => {
    const res = await fetch(`${base}/v1/messages`, {
      method: 'POST', headers,
      body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }], max_tokens: 5, stream: false }),
    })
    expect(res.status).toBe(200)
    const body = await res.json() as { type: string; content: Array<{ type: string; text: string }> }
    expect(body.type).toBe('message')
    expect(body.content[0].type).toBe('text')
  })

  it('streams anthropic SSE events', async () => {
    const res = await fetch(`${base}/v1/messages`, {
      method: 'POST', headers,
      body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }], max_tokens: 5, stream: true }),
    })
    expect(res.status).toBe(200)
    const text = await res.text()
    expect(text).toContain('message_start')
    expect(text).toContain('message_stop')
  })
})
