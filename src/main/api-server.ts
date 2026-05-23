import { Hono } from 'hono'
import { serve } from '@hono/node-server'
import { streamSSE } from 'hono/streaming'
import { randomUUID } from 'crypto'
import type { Server } from 'node:http'
import type { AsyncBlockRunner } from '../inference/native-worker'
import type { NativeTokenizer } from '../inference/native-tokenizer'
import type { ModelInfo } from './model-manager'
import type { ApiServerConfig } from './api-config'
import { runInferenceStream } from './inference-orchestrator'
import { ChatTemplateUnavailableError } from '../inference/native-tokenizer'

export interface ApiServerDeps {
  getRunner: () => AsyncBlockRunner | null
  getTokenizer: () => NativeTokenizer | null
  getModel: () => ModelInfo | null
}

export interface LogEntry {
  ts: number
  method: string
  path: string
  status: number
  ms: number
}

export class ApiServer {
  private server: Server | null = null
  private config: ApiServerConfig
  private readonly deps: ApiServerDeps
  private logListeners: ((entry: LogEntry) => void)[] = []

  constructor(config: ApiServerConfig, deps: ApiServerDeps) {
    this.config = config
    this.deps = deps
  }

  get running(): boolean { return this.server !== null }
  get port(): number { return this.config.port }
  get apiKey(): string { return this.config.apiKey }

  onLog(listener: (entry: LogEntry) => void): () => void {
    this.logListeners.push(listener)
    return () => { this.logListeners = this.logListeners.filter(l => l !== listener) }
  }

  updateConfig(cfg: ApiServerConfig): void {
    this.config = cfg
  }

  async start(): Promise<void> {
    if (this.server) return
    const app = buildApp(() => this.config.apiKey, this.deps, this.config.port, (e) => this.logListeners.forEach(l => l(e)))
    try {
      await new Promise<void>((resolve, reject) => {
        this.server = serve({ fetch: app.fetch, port: this.config.port }, () => resolve()) as Server
        this.server.once('error', reject)
      })
    } catch (err) {
      this.server = null
      throw err
    }
  }

  async stop(): Promise<void> {
    if (!this.server) return
    await new Promise<void>((resolve, reject) => {
      this.server!.close((err) => (err ? reject(err) : resolve()))
    })
    this.server = null
  }
}

type ContentBlock = { type: string; text?: string }

function normalizeContent(content: unknown): string {
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return (content as ContentBlock[])
      .filter(b => b.type === 'text' && typeof b.text === 'string')
      .map(b => b.text as string)
      .join('')
  }
  return String(content ?? '')
}

function buildApp(getKey: () => string, deps: ApiServerDeps, port: number, onLog?: (entry: LogEntry) => void): Hono {
  const app = new Hono()

  app.onError((err, c) => {
    const msg = err instanceof Error ? err.message : String(err)
    const isJsonParse = msg.includes('is not valid JSON') || msg.includes('JSON Parse') || msg.includes('Unexpected token')
    if (isJsonParse) {
      return c.json({ error: { message: 'Invalid JSON in request body' } }, 400)
    }
    console.error('[ApiServer] Unhandled error:', err)
    return c.json({ error: { message: msg } }, 500)
  })

  app.use('*', async (c, next) => {
    const auth = c.req.header('Authorization')
    if (auth !== `Bearer ${getKey()}`) {
      onLog?.({ ts: Date.now(), method: c.req.method, path: new URL(c.req.url).pathname, status: 401, ms: 0 })
      return c.json({ error: { message: 'Invalid API key' } }, 401)
    }
    const t0 = Date.now()
    await next()
    onLog?.({ ts: t0, method: c.req.method, path: new URL(c.req.url).pathname, status: c.res.status, ms: Date.now() - t0 })
  })

  app.get('/health', (c) => {
    const model = deps.getModel()
    return c.json({ status: 'ok', model: model?.repoId ?? null, port })
  })

  app.get('/v1/models', (c) => {
    const model = deps.getModel()
    return c.json({
      object: 'list',
      data: model
        ? [{ id: model.repoId ?? 'opencoral', object: 'model', created: 0, owned_by: 'opencoral' }]
        : [],
    })
  })

  app.post('/v1/chat/completions', async (c) => {
    const runner = deps.getRunner()
    const tokenizer = deps.getTokenizer()
    if (!runner || !tokenizer) {
      return c.json({ error: { message: 'No model loaded' } }, 503)
    }

    const body = await c.req.json() as {
      messages?: Array<{ role: string; content: string | unknown[] }>
      max_tokens?: number
      stream?: boolean
    }
    const messages = body.messages ?? []
    const maxTokens = body.max_tokens ?? 512
    const stream = body.stream ?? false
    const requestId = randomUUID()

    let promptIds: Int32Array
    try {
      const turns = messages.map(m => ({
        role: m.role as 'user' | 'assistant' | 'system',
        content: normalizeContent(m.content),
      }))
      promptIds = await tokenizer.encodeChatMulti(turns)
    } catch (err) {
      if (err instanceof ChatTemplateUnavailableError) {
        return c.json({ error: { message: 'Model has no chat template' } }, 400)
      }
      throw err
    }

    const modelId = deps.getModel()?.repoId ?? 'opencoral'
    const created = Math.floor(Date.now() / 1000)

    if (!stream) {
      const chunks: string[] = []
      for await (const piece of runInferenceStream({ runner, tokenizer, promptIds, maxTokens, requestId })) {
        chunks.push(piece)
      }
      return c.json({
        id: `chatcmpl-${requestId}`,
        object: 'chat.completion',
        created,
        model: modelId,
        choices: [{
          index: 0,
          message: { role: 'assistant', content: chunks.join('') },
          finish_reason: 'stop',
        }],
        usage: { prompt_tokens: promptIds.length, completion_tokens: chunks.length, total_tokens: promptIds.length + chunks.length },
      })
    }

    return streamSSE(c, async (sseStream) => {
      const id = `chatcmpl-${requestId}`
      for await (const piece of runInferenceStream({ runner, tokenizer, promptIds, maxTokens, requestId })) {
        await sseStream.writeSSE({
          data: JSON.stringify({
            id, object: 'chat.completion.chunk', created, model: modelId,
            choices: [{ index: 0, delta: { content: piece }, finish_reason: null }],
          }),
        })
      }
      await sseStream.writeSSE({
        data: JSON.stringify({
          id, object: 'chat.completion.chunk', created, model: modelId,
          choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
        }),
      })
      await sseStream.writeSSE({ data: '[DONE]' })
    })
  })

  app.post('/v1/messages', async (c) => {
    const runner = deps.getRunner()
    const tokenizer = deps.getTokenizer()
    if (!runner || !tokenizer) {
      return c.json({ error: { type: 'error', message: 'No model loaded' } }, 503)
    }

    const body = await c.req.json() as {
      messages?: Array<{ role: string; content: string | unknown[] }>
      max_tokens?: number
      stream?: boolean
    }
    const messages = body.messages ?? []
    const maxTokens = body.max_tokens ?? 512
    const stream = body.stream ?? false
    const requestId = randomUUID()

    let promptIds: Int32Array
    try {
      const turns = messages.map(m => ({
        role: m.role as 'user' | 'assistant' | 'system',
        content: normalizeContent(m.content),
      }))
      promptIds = await tokenizer.encodeChatMulti(turns)
    } catch (err) {
      if (err instanceof ChatTemplateUnavailableError) {
        return c.json({ error: { type: 'error', message: 'Model has no chat template' } }, 400)
      }
      throw err
    }

    const modelId = deps.getModel()?.repoId ?? 'opencoral'
    const msgId = `msg_${requestId.replace(/-/g, '').slice(0, 24)}`

    if (!stream) {
      const chunks: string[] = []
      for await (const piece of runInferenceStream({ runner, tokenizer, promptIds, maxTokens, requestId })) {
        chunks.push(piece)
      }
      return c.json({
        id: msgId,
        type: 'message',
        role: 'assistant',
        content: [{ type: 'text', text: chunks.join('') }],
        model: modelId,
        stop_reason: 'end_turn',
        stop_sequence: null,
        usage: { input_tokens: promptIds.length, output_tokens: chunks.length },
      })
    }

    return streamSSE(c, async (sseStream) => {
      const write = async (event: string, data: unknown) => {
        await sseStream.writeSSE({ event, data: JSON.stringify(data) })
      }

      await write('message_start', {
        type: 'message_start',
        message: { id: msgId, type: 'message', role: 'assistant', content: [], model: modelId },
      })
      await write('content_block_start', {
        type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' },
      })

      let outputTokens = 0
      for await (const piece of runInferenceStream({ runner, tokenizer, promptIds, maxTokens, requestId })) {
        await write('content_block_delta', {
          type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece },
        })
        outputTokens++
      }

      await write('content_block_stop', { type: 'content_block_stop', index: 0 })
      await write('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: 'end_turn', stop_sequence: null },
        usage: { output_tokens: outputTokens },
      })
      await write('message_stop', { type: 'message_stop' })
    })
  })

  return app
}
