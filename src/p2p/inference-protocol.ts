import type { Libp2p } from 'libp2p'
import type { PeerId, Stream } from '@libp2p/interface'

export const INFERENCE_PROTOCOL = '/opencoral/inference/1.0.0'

export type InferenceHandler = (
  input: Float32Array,
  nTokens: number,
  nEmbd: number,
) => Promise<Float32Array>

// Wire format: [uint32 LE: nTokens][uint32 LE: nEmbd][float32 LE × nTokens × nEmbd]

function encodeMessage(data: Float32Array, nTokens: number, nEmbd: number): Uint8Array {
  const buf = Buffer.allocUnsafe(8 + data.byteLength)
  buf.writeUInt32LE(nTokens, 0)
  buf.writeUInt32LE(nEmbd, 4)
  Buffer.from(data.buffer, data.byteOffset, data.byteLength).copy(buf, 8)
  return buf
}

function decodeMessage(buf: Buffer): { nTokens: number; nEmbd: number; data: Float32Array } {
  if (buf.length < 8) {
    throw new Error(`Message header truncated: expected 8 bytes, got ${buf.length}`)
  }
  const nTokens = buf.readUInt32LE(0)
  const nEmbd = buf.readUInt32LE(4)
  const expectedBytes = nTokens * nEmbd * 4
  if (buf.length < 8 + expectedBytes) {
    throw new Error(`Message payload truncated: expected ${8 + expectedBytes} bytes, got ${buf.length}`)
  }
  const floatByteLength = nTokens * nEmbd * 4
  const floatBuf = Buffer.allocUnsafe(floatByteLength)
  buf.copy(floatBuf, 0, 8, 8 + floatByteLength)
  return { nTokens, nEmbd, data: new Float32Array(floatBuf.buffer) }
}

async function collectStream(stream: AsyncIterable<Uint8Array | { subarray(): Uint8Array }>): Promise<Buffer> {
  const chunks: Buffer[] = []
  for await (const chunk of stream) {
    // libp2p streams yield Uint8ArrayList (has .subarray()) or plain Uint8Array
    const bytes: Uint8Array = typeof (chunk as any).subarray === 'function'
      ? (chunk as any).subarray()
      : (chunk as Uint8Array)
    chunks.push(Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength))
  }
  return Buffer.concat(chunks)
}

async function sendWithBackpressure(stream: Stream, data: Uint8Array): Promise<void> {
  if (!stream.send(data)) {
    // send() returns false when the write buffer is full; wait for drain before continuing
    await new Promise<void>(resolve => {
      stream.addEventListener('drain', () => resolve(), { once: true })
    })
  }
}

export async function registerInferenceHandler(
  libp2p: Libp2p,
  handler: InferenceHandler,
): Promise<void> {
  await libp2p.handle(INFERENCE_PROTOCOL, async (stream) => {
    try {
      const buf = await collectStream(stream)
      const { nTokens, nEmbd, data } = decodeMessage(buf)
      const output = await handler(data, nTokens, nEmbd)
      const response = encodeMessage(output, nTokens, nEmbd)
      await sendWithBackpressure(stream, response)
      await stream.close()
    } catch (err) {
      stream.abort(err instanceof Error ? err : new Error(String(err)))
    }
  }, { force: true })
}

export async function sendInferenceRequest(
  libp2p: Libp2p,
  peerId: PeerId,
  input: Float32Array,
  nTokens: number,
  nEmbd: number,
): Promise<Float32Array> {
  const stream = await libp2p.dialProtocol(peerId, INFERENCE_PROTOCOL)
  const request = encodeMessage(input, nTokens, nEmbd)
  await sendWithBackpressure(stream, request)
  // Half-close the write side concurrently with reading the response,
  // so the remote receives EOF without a sequential ordering race
  const [buf] = await Promise.all([collectStream(stream), stream.close()])
  return decodeMessage(buf).data
}
