import type { Libp2p, PeerId, Stream } from '@libp2p/interface'

export const KV_PROTOCOL = '/opencoral/kv/1.0.0'

const MSG_OPEN    = 0x01
const MSG_FORWARD = 0x02
const MSG_CLOSE   = 0x03
const STATUS_OK   = 0x00
const STATUS_ERR  = 0x01

export interface KVSessionHandler {
  onOpen(sessionId: string, maxSeqLen: number): Promise<{ ok: true }>
  onForward(sessionId: string, input: Float32Array, nTokens: number, nEmbd: number): Promise<Float32Array>
  onClose(sessionId: string): Promise<void>
}

// ── Stream helpers ────────────────────────────────────────────────────────────

async function collectStream(stream: AsyncIterable<Uint8Array | { subarray(): Uint8Array }>): Promise<Buffer> {
  const chunks: Buffer[] = []
  for await (const chunk of stream) {
    const bytes: Uint8Array = typeof (chunk as any).subarray === 'function'
      ? (chunk as any).subarray()
      : (chunk as Uint8Array)
    chunks.push(Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength))
  }
  return Buffer.concat(chunks)
}

async function sendWithBackpressure(stream: Stream, data: Uint8Array): Promise<void> {
  if (!stream.send(data)) {
    await new Promise<void>(resolve => {
      stream.addEventListener('drain', () => resolve(), { once: true })
    })
  }
}

// ── Message encoding ──────────────────────────────────────────────────────────

function encodeRequest(msgType: number, json: object, tensor?: Float32Array): Buffer {
  const jsonBytes = Buffer.from(JSON.stringify(json), 'utf-8')
  const jsonLenBuf = Buffer.allocUnsafe(4)
  jsonLenBuf.writeUInt32LE(jsonBytes.byteLength, 0)
  const parts: Buffer[] = [Buffer.from([msgType]), jsonLenBuf, jsonBytes]
  if (tensor) {
    parts.push(Buffer.from(tensor.buffer, tensor.byteOffset, tensor.byteLength))
  }
  return Buffer.concat(parts)
}

function decodeRequest(buf: Buffer): { msgType: number; json: any; tensorBytes?: Buffer } {
  const msgType = buf[0]
  const jsonLen = buf.readUInt32LE(1)
  const json = JSON.parse(buf.slice(5, 5 + jsonLen).toString('utf-8'))
  const tensorBytes = buf.byteLength > 5 + jsonLen ? buf.slice(5 + jsonLen) : undefined
  return { msgType, json, tensorBytes }
}

function encodeOkResponse(): Buffer {
  const body = Buffer.from('{"ok":true}', 'utf-8')
  return Buffer.concat([Buffer.from([STATUS_OK]), body])
}

function encodeErrorResponse(msg: string): Buffer {
  const body = Buffer.from(JSON.stringify({ error: msg }), 'utf-8')
  return Buffer.concat([Buffer.from([STATUS_ERR]), body])
}

function encodeTensorResponse(tensor: Float32Array): Buffer {
  // Prefix with STATUS_OK so the client can distinguish success from error
  const tensorBuf = Buffer.from(tensor.buffer, tensor.byteOffset, tensor.byteLength)
  return Buffer.concat([Buffer.from([STATUS_OK]), tensorBuf])
}

// ── Handler registration ──────────────────────────────────────────────────────

export async function registerKVHandler(libp2p: Libp2p, handler: KVSessionHandler): Promise<void> {
  await libp2p.handle(KV_PROTOCOL, async (stream) => {
    try {
      const buf = await collectStream(stream)
      const { msgType, json, tensorBytes } = decodeRequest(buf)

      let response: Buffer

      if (msgType === MSG_OPEN) {
        const result = await handler.onOpen(json.sessionId as string, json.maxSeqLen as number)
        response = result.ok ? encodeOkResponse() : encodeErrorResponse('onOpen rejected')
      } else if (msgType === MSG_FORWARD) {
        const { sessionId, nTokens, nEmbd } = json as { sessionId: string; nTokens: number; nEmbd: number }
        if (!tensorBytes) throw new Error('FORWARD message missing tensor bytes')
        // Copy into aligned buffer — tensorBytes may have non-multiple-of-4 byteOffset
        const alignedBuf = Buffer.allocUnsafe(tensorBytes.byteLength)
        tensorBytes.copy(alignedBuf, 0)
        const input = new Float32Array(alignedBuf.buffer, 0, tensorBytes.byteLength / 4)
        const output = await handler.onForward(sessionId, input, nTokens, nEmbd)
        response = encodeTensorResponse(output)
      } else if (msgType === MSG_CLOSE) {
        await handler.onClose(json.sessionId as string)
        response = encodeOkResponse()
      } else {
        response = encodeErrorResponse(`Unknown message type: ${msgType}`)
      }

      await sendWithBackpressure(stream, response)
      await stream.close()
    } catch (err) {
      stream.abort(err instanceof Error ? err : new Error(String(err)))
    }
  }, { force: true })
}

// ── Client functions ──────────────────────────────────────────────────────────

export async function openRemoteSession(
  libp2p: Libp2p,
  peerId: PeerId,
  sessionId: string,
  maxSeqLen: number,
): Promise<void> {
  const stream = await libp2p.dialProtocol(peerId, KV_PROTOCOL, { signal: AbortSignal.timeout(5000) })
  const req = encodeRequest(MSG_OPEN, { sessionId, maxSeqLen })
  await sendWithBackpressure(stream, req)
  const [buf] = await Promise.all([collectStream(stream), stream.close().catch(() => {})])
  if (buf[0] === STATUS_ERR) {
    throw new Error(`KV open failed: ${buf.slice(1).toString('utf-8')}`)
  }
}

export async function forwardRemote(
  libp2p: Libp2p,
  peerId: PeerId,
  sessionId: string,
  input: Float32Array,
  nTokens: number,
  nEmbd: number,
): Promise<Float32Array> {
  const stream = await libp2p.dialProtocol(peerId, KV_PROTOCOL, { signal: AbortSignal.timeout(10000) })
  const req = encodeRequest(MSG_FORWARD, { sessionId, nTokens, nEmbd }, input)
  await sendWithBackpressure(stream, req)
  const [buf] = await Promise.all([collectStream(stream), stream.close().catch(() => {})])

  if (buf.byteLength === 0) throw new Error('KV forward: empty response')
  if (buf[0] === STATUS_ERR) {
    throw new Error(`KV forward failed: ${buf.slice(1).toString('utf-8')}`)
  }
  // Success: STATUS_OK byte followed by raw float32 bytes — copy to ensure alignment
  const tensorBytes = buf.slice(1)
  const alignedBuf = Buffer.allocUnsafe(tensorBytes.byteLength)
  tensorBytes.copy(alignedBuf, 0)
  return new Float32Array(alignedBuf.buffer, 0, tensorBytes.byteLength / 4)
}

export async function closeRemoteSession(
  libp2p: Libp2p,
  peerId: PeerId,
  sessionId: string,
): Promise<void> {
  const stream = await libp2p.dialProtocol(peerId, KV_PROTOCOL, { signal: AbortSignal.timeout(3000) })
  const req = encodeRequest(MSG_CLOSE, { sessionId })
  await sendWithBackpressure(stream, req)
  await Promise.all([collectStream(stream), stream.close().catch(() => {})])
}
