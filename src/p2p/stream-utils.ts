import type { Stream } from '@libp2p/interface'

export const DEFAULT_CHUNK_SIZE = 64 * 1024

export async function collectStream(
  stream: AsyncIterable<Uint8Array | { subarray(): Uint8Array }>,
): Promise<Buffer> {
  const chunks: Buffer[] = []
  for await (const chunk of stream) {
    const bytes: Uint8Array = typeof (chunk as any).subarray === 'function'
      ? (chunk as any).subarray()
      : (chunk as Uint8Array)
    chunks.push(Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength))
  }
  return Buffer.concat(chunks)
}

export async function sendWithBackpressure(stream: Stream, data: Uint8Array): Promise<void> {
  if (!stream.send(data)) {
    await new Promise<void>(resolve => {
      stream.addEventListener('drain', () => resolve(), { once: true })
    })
  }
}

export function encodeChunked(data: Uint8Array, chunkSize: number = DEFAULT_CHUNK_SIZE): Buffer[] {
  const frames: Buffer[] = []
  let offset = 0
  while (offset < data.byteLength) {
    const end = Math.min(offset + chunkSize, data.byteLength)
    const frameData = data.slice(offset, end)
    const header = Buffer.allocUnsafe(4)
    header.writeUInt32LE(frameData.byteLength, 0)
    frames.push(Buffer.concat([header, Buffer.from(frameData)]))
    offset = end
  }
  const sentinel = Buffer.allocUnsafe(4)
  sentinel.writeUInt32LE(0, 0)
  frames.push(sentinel)
  return frames
}

export function decodeChunked(frames: Buffer[]): Buffer {
  const chunks: Buffer[] = []
  for (const frame of frames) {
    const size = frame.readUInt32LE(0)
    if (size === 0) break
    chunks.push(frame.slice(4, 4 + size))
  }
  return Buffer.concat(chunks)
}

export async function collectChunkedStream(
  stream: AsyncIterable<Uint8Array | { subarray(): Uint8Array }>,
): Promise<Buffer> {
  const rawBuf = await collectStream(stream)
  const frames: Buffer[] = []
  let pos = 0
  while (pos + 4 <= rawBuf.byteLength) {
    const size = rawBuf.readUInt32LE(pos)
    frames.push(rawBuf.slice(pos, pos + 4 + size))
    pos += 4 + size
    if (size === 0) break
  }
  return decodeChunked(frames)
}

export async function sendChunked(stream: Stream, data: Uint8Array): Promise<void> {
  for (const frame of encodeChunked(data)) {
    await sendWithBackpressure(stream, frame)
  }
}

export class FramedReader {
  private buffer: Buffer = Buffer.alloc(0)
  private done = false
  private iterator: AsyncIterator<Uint8Array | { subarray(): Uint8Array }>

  constructor(stream: AsyncIterable<Uint8Array | { subarray(): Uint8Array }>) {
    this.iterator = stream[Symbol.asyncIterator]()
  }

  async readExact(n: number): Promise<Buffer> {
    while (this.buffer.byteLength < n) {
      if (this.done) throw new Error(`FramedReader: unexpected end of stream (needed ${n} bytes, have ${this.buffer.byteLength})`)
      const result = await this.iterator.next()
      if (result.done) {
        this.done = true
        continue
      }
      const chunk = result.value
      const bytes: Uint8Array = typeof (chunk as any).subarray === 'function'
        ? (chunk as any).subarray()
        : (chunk as Uint8Array)
      this.buffer = Buffer.concat([this.buffer, Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength)])
    }
    const out = this.buffer.slice(0, n)
    this.buffer = this.buffer.slice(n)
    return out
  }

  async readUint8(): Promise<number> {
    const buf = await this.readExact(1)
    return buf[0]
  }

  async readUint16LE(): Promise<number> {
    const buf = await this.readExact(2)
    return buf.readUInt16LE(0)
  }

  async readUint32LE(): Promise<number> {
    const buf = await this.readExact(4)
    return buf.readUInt32LE(0)
  }
}
