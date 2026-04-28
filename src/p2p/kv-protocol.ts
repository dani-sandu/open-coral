import type { Libp2p, PeerId, Stream } from '@libp2p/interface'
import {
  sendWithBackpressure,
  encodeChunked,
  FramedReader,
  DEFAULT_CHUNK_SIZE,
} from './stream-utils'

export const KV_PROTOCOL = '/opencoral/kv/2.0.0'

const KV_IDLE_TIMEOUT_MS = 5 * 60 * 1000 // 5 minutes

const MSG_OPEN    = 0x01
const MSG_FORWARD = 0x02
const MSG_CLOSE     = 0x03
const MSG_ROLLBACK  = 0x04
const MSG_FORWARD_ALL = 0x05
const STATUS_OK   = 0x00
const STATUS_ERR  = 0x01

const FLAG_RAW     = 0x00
const FLAG_CHUNKED = 0x01

export interface KVSessionHandler {
  onOpen(sessionId: string, maxSeqLen: number): Promise<{ ok: true }>
  onForward(sessionId: string, input: Float32Array, nTokens: number, nEmbd: number, requestId?: string): Promise<Float32Array>
  onClose(sessionId: string): Promise<void>
  onRollback?(sessionId: string, newNPast: number): Promise<void>
  /** Batch logit retrieval for speculative decoding verification. Returns nTokens × vocabSize floats.
   *  requestId is intentionally omitted — batch calls are not individually traced. */
  onForwardAll?(sessionId: string, input: Float32Array, nTokens: number, nEmbd: number): Promise<Float32Array>
}

// ── Writing helpers ──────────────────────────────────────────────────────────

async function writeRawMessage(stream: Stream, msgType: number, payload: Buffer): Promise<void> {
  const header = Buffer.allocUnsafe(2 + 4)
  header.writeUInt8(msgType, 0)
  header.writeUInt8(FLAG_RAW, 1)
  header.writeUInt32LE(payload.byteLength, 2)
  await sendWithBackpressure(stream, header)
  if (payload.byteLength > 0) {
    await sendWithBackpressure(stream, payload)
  }
}

async function writeChunkedMessage(stream: Stream, msgType: number, payload: Buffer): Promise<void> {
  const header = Buffer.allocUnsafe(2)
  header.writeUInt8(msgType, 0)
  header.writeUInt8(FLAG_CHUNKED, 1)
  await sendWithBackpressure(stream, header)
  for (const frame of encodeChunked(payload)) {
    await sendWithBackpressure(stream, frame)
  }
}

async function writeResponse(stream: Stream, status: number, payload: Buffer, chunked: boolean): Promise<void> {
  const header = Buffer.allocUnsafe(2)
  header.writeUInt8(status, 0)
  header.writeUInt8(chunked ? FLAG_CHUNKED : FLAG_RAW, 1)
  await sendWithBackpressure(stream, header)
  if (chunked) {
    for (const frame of encodeChunked(payload)) {
      await sendWithBackpressure(stream, frame)
    }
  } else {
    const lenBuf = Buffer.allocUnsafe(4)
    lenBuf.writeUInt32LE(payload.byteLength, 0)
    await sendWithBackpressure(stream, lenBuf)
    if (payload.byteLength > 0) {
      await sendWithBackpressure(stream, payload)
    }
  }
}

// ── Reading helpers ──────────────────────────────────────────────────────────

async function readPayload(reader: FramedReader, flags: number): Promise<Buffer> {
  if (flags & FLAG_CHUNKED) {
    const chunks: Buffer[] = []
    while (true) {
      const size = await reader.readUint32LE()
      if (size === 0) break
      chunks.push(await reader.readExact(size))
    }
    return Buffer.concat(chunks)
  } else {
    const len = await reader.readUint32LE()
    if (len === 0) return Buffer.alloc(0)
    return reader.readExact(len)
  }
}

// ── Handler registration ─────────────────────────────────────────────────────

export async function registerKVHandler(libp2p: Libp2p, handler: KVSessionHandler): Promise<void> {
  await libp2p.handle(KV_PROTOCOL, async (stream) => {
    const reader = new FramedReader(stream)
    let idleTimer: ReturnType<typeof setTimeout> | null = null

    function resetIdleTimeout(): Promise<never> {
      return new Promise((_, reject) => {
        if (idleTimer) clearTimeout(idleTimer)
        idleTimer = setTimeout(() => reject(new Error('KV session idle timeout')), KV_IDLE_TIMEOUT_MS)
      })
    }

    try {
      while (true) {
        const timeoutPromise = resetIdleTimeout()
        const msgType = await Promise.race([reader.readUint8(), timeoutPromise])
        const flags = await reader.readUint8()
        const payload = await readPayload(reader, flags)

        if (msgType === MSG_OPEN) {
          const json = JSON.parse(payload.toString('utf-8'))
          await handler.onOpen(json.sessionId as string, json.maxSeqLen as number)
          await writeResponse(stream, STATUS_OK, Buffer.from('{"ok":true}', 'utf-8'), false)
        } else if (msgType === MSG_FORWARD) {
          const jsonLen = payload.readUInt32LE(0)
          const json = JSON.parse(payload.slice(4, 4 + jsonLen).toString('utf-8'))
          const { sessionId, nTokens, nEmbd, requestId } = json
          const tensorBytes = payload.slice(4 + jsonLen)
          const alignedBuf = Buffer.allocUnsafe(tensorBytes.byteLength)
          tensorBytes.copy(alignedBuf, 0)
          const input = new Float32Array(alignedBuf.buffer, 0, tensorBytes.byteLength / 4)
          const output = await handler.onForward(sessionId, input, nTokens, nEmbd, requestId)
          const tensorOut = Buffer.from(output.buffer, output.byteOffset, output.byteLength)
          const useChunked = tensorOut.byteLength > DEFAULT_CHUNK_SIZE
          await writeResponse(stream, STATUS_OK, tensorOut, useChunked)
        } else if (msgType === MSG_CLOSE) {
          const json = JSON.parse(payload.toString('utf-8'))
          await handler.onClose(json.sessionId as string)
          await writeResponse(stream, STATUS_OK, Buffer.from('{"ok":true}', 'utf-8'), false)
          break
        } else if (msgType === MSG_ROLLBACK) {
          const json = JSON.parse(payload.toString('utf-8'))
          if (handler.onRollback) {
            await handler.onRollback(json.sessionId as string, json.newNPast as number)
          }
          await writeResponse(stream, STATUS_OK, Buffer.from('{"ok":true}', 'utf-8'), false)
        } else if (msgType === MSG_FORWARD_ALL) {
          const jsonLen = payload.readUInt32LE(0)
          const json = JSON.parse(payload.slice(4, 4 + jsonLen).toString('utf-8'))
          const { sessionId, nTokens, nEmbd } = json
          const tensorBytes = payload.slice(4 + jsonLen)
          const alignedBuf = Buffer.allocUnsafe(tensorBytes.byteLength)
          tensorBytes.copy(alignedBuf, 0)
          const input = new Float32Array(alignedBuf.buffer, 0, tensorBytes.byteLength / 4)
          if (!handler.onForwardAll) {
            // Respond with error but keep the stream alive so other ops (close, rollback) still work.
            await writeResponse(stream, STATUS_ERR, Buffer.from('onForwardAll not implemented', 'utf-8'), false)
          } else {
            const output = await handler.onForwardAll(sessionId, input, nTokens, nEmbd)
            const tensorOut = Buffer.from(output.buffer, output.byteOffset, output.byteLength)
            const useChunked = tensorOut.byteLength > DEFAULT_CHUNK_SIZE
            await writeResponse(stream, STATUS_OK, tensorOut, useChunked)
          }
        } else {
          await writeResponse(stream, STATUS_ERR, Buffer.from(`Unknown msg type: ${msgType}`, 'utf-8'), false)
          break
        }
      }
      if (idleTimer) clearTimeout(idleTimer)
      await stream.close()
    } catch (err) {
      if (idleTimer) clearTimeout(idleTimer)
      stream.abort(err instanceof Error ? err : new Error(String(err)))
    }
  }, { force: true })
}

// ── Client ───────────────────────────────────────────────────────────────────

export class KVSessionClient {
  private readonly stream: Stream
  private readonly reader: FramedReader
  private readonly sessionId: string

  private constructor(stream: Stream, sessionId: string) {
    this.stream = stream
    this.reader = new FramedReader(stream)
    this.sessionId = sessionId
  }

  static async open(
    libp2p: Libp2p,
    peerId: PeerId,
    sessionId: string,
    maxSeqLen: number,
  ): Promise<KVSessionClient> {
    const stream = await libp2p.dialProtocol(peerId, KV_PROTOCOL, { signal: AbortSignal.timeout(5000) })
    const client = new KVSessionClient(stream, sessionId)

    const json = Buffer.from(JSON.stringify({ sessionId, maxSeqLen }), 'utf-8')
    await writeRawMessage(stream, MSG_OPEN, json)

    const status = await client.reader.readUint8()
    const flags = await client.reader.readUint8()
    await readPayload(client.reader, flags)
    if (status === STATUS_ERR) throw new Error(`KV open failed for session ${sessionId}`)

    return client
  }

  async forward(
    input: Float32Array,
    nTokens: number,
    nEmbd: number,
    requestId?: string,
  ): Promise<Float32Array> {
    const jsonObj: any = { sessionId: this.sessionId, nTokens, nEmbd }
    if (requestId) jsonObj.requestId = requestId
    const jsonBytes = Buffer.from(JSON.stringify(jsonObj), 'utf-8')
    const jsonLenBuf = Buffer.allocUnsafe(4)
    jsonLenBuf.writeUInt32LE(jsonBytes.byteLength, 0)
    const tensorBuf = Buffer.from(input.buffer, input.byteOffset, input.byteLength)
    const payload = Buffer.concat([jsonLenBuf, jsonBytes, tensorBuf])

    const useChunked = payload.byteLength > DEFAULT_CHUNK_SIZE
    if (useChunked) {
      await writeChunkedMessage(this.stream, MSG_FORWARD, payload)
    } else {
      await writeRawMessage(this.stream, MSG_FORWARD, payload)
    }

    const status = await this.reader.readUint8()
    const flags = await this.reader.readUint8()
    const respPayload = await readPayload(this.reader, flags)

    if (status === STATUS_ERR) throw new Error(`KV forward failed: ${respPayload.toString('utf-8')}`)

    const alignedBuf = Buffer.allocUnsafe(respPayload.byteLength)
    respPayload.copy(alignedBuf, 0)
    return new Float32Array(alignedBuf.buffer, 0, respPayload.byteLength / 4)
  }

  async forwardAll(input: Float32Array, nTokens: number, nEmbd: number): Promise<Float32Array> {
    const jsonObj = { sessionId: this.sessionId, nTokens, nEmbd }
    const jsonBytes = Buffer.from(JSON.stringify(jsonObj), 'utf-8')
    const jsonLenBuf = Buffer.allocUnsafe(4)
    jsonLenBuf.writeUInt32LE(jsonBytes.byteLength, 0)
    const tensorBuf = Buffer.from(input.buffer, input.byteOffset, input.byteLength)
    const payload = Buffer.concat([jsonLenBuf, jsonBytes, tensorBuf])

    const useChunked = payload.byteLength > DEFAULT_CHUNK_SIZE
    if (useChunked) {
      await writeChunkedMessage(this.stream, MSG_FORWARD_ALL, payload)
    } else {
      await writeRawMessage(this.stream, MSG_FORWARD_ALL, payload)
    }

    const status = await this.reader.readUint8()
    const flags = await this.reader.readUint8()
    const respPayload = await readPayload(this.reader, flags)

    if (status === STATUS_ERR) throw new Error(`KV forwardAll failed: ${respPayload.toString('utf-8')}`)

    const alignedBuf = Buffer.allocUnsafe(respPayload.byteLength)
    respPayload.copy(alignedBuf, 0)
    return new Float32Array(alignedBuf.buffer, 0, respPayload.byteLength / 4)
  }

  async rollback(newNPast: number): Promise<void> {
    const json = Buffer.from(JSON.stringify({ sessionId: this.sessionId, newNPast }), 'utf-8')
    await writeRawMessage(this.stream, MSG_ROLLBACK, json)

    const status = await this.reader.readUint8()
    const flags = await this.reader.readUint8()
    await readPayload(this.reader, flags)
    if (status === STATUS_ERR) throw new Error(`KV rollback failed for session ${this.sessionId}`)
  }

  async close(): Promise<void> {
    const json = Buffer.from(JSON.stringify({ sessionId: this.sessionId }), 'utf-8')
    await writeRawMessage(this.stream, MSG_CLOSE, json)

    const status = await this.reader.readUint8()
    const flags = await this.reader.readUint8()
    await readPayload(this.reader, flags)
    if (status === STATUS_ERR) console.warn(`KV close returned error for session ${this.sessionId}`)

    await this.stream.close()
  }
}
