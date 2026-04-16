import { describe, it, expect } from 'bun:test'
import {
  collectStream,
  encodeChunked,
  decodeChunked,
  DEFAULT_CHUNK_SIZE,
  FramedReader,
} from '../../src/p2p/stream-utils'

describe('stream-utils', () => {
  describe('encodeChunked / decodeChunked', () => {
    it('roundtrip with small data', () => {
      const data = new Uint8Array([1, 2, 3, 4, 5])
      const frames = encodeChunked(data, 3)
      expect(frames).toHaveLength(3)
      const result = decodeChunked(frames)
      expect(Array.from(result)).toEqual([1, 2, 3, 4, 5])
    })

    it('roundtrip with data fitting in one chunk', () => {
      const data = new Uint8Array([7, 8, 9])
      const frames = encodeChunked(data, 1024)
      const result = decodeChunked(frames)
      expect(Array.from(result)).toEqual([7, 8, 9])
    })

    it('empty input produces sentinel only', () => {
      const frames = encodeChunked(Buffer.alloc(0), 64)
      const result = decodeChunked(frames)
      expect(result.byteLength).toBe(0)
    })
  })

  describe('collectStream', () => {
    it('collects async iterable into single Buffer', async () => {
      async function* gen() {
        yield new Uint8Array([1, 2])
        yield new Uint8Array([3, 4, 5])
      }
      const buf = await collectStream(gen())
      expect(Array.from(buf)).toEqual([1, 2, 3, 4, 5])
    })

    it('returns empty buffer for empty stream', async () => {
      async function* gen() {}
      const buf = await collectStream(gen())
      expect(buf.byteLength).toBe(0)
    })
  })

  it('DEFAULT_CHUNK_SIZE is 64KB', () => {
    expect(DEFAULT_CHUNK_SIZE).toBe(64 * 1024)
  })

  describe('FramedReader', () => {
    it('reads exact number of bytes across chunk boundaries', async () => {
      async function* gen() {
        yield new Uint8Array([1, 2, 3])
        yield new Uint8Array([4, 5])
        yield new Uint8Array([6, 7, 8, 9, 10])
      }
      const reader = new FramedReader(gen())
      const first = await reader.readExact(4)
      expect(Array.from(first)).toEqual([1, 2, 3, 4])
      const second = await reader.readExact(3)
      expect(Array.from(second)).toEqual([5, 6, 7])
    })

    it('reads uint8', async () => {
      async function* gen() { yield new Uint8Array([42]) }
      const reader = new FramedReader(gen())
      expect(await reader.readUint8()).toBe(42)
    })

    it('reads uint32LE across chunk boundary', async () => {
      async function* gen() {
        yield new Uint8Array([0x02, 0x01])
        yield new Uint8Array([0x00, 0x00])
      }
      const reader = new FramedReader(gen())
      expect(await reader.readUint32LE()).toBe(258)
    })

    it('throws when stream ends before enough bytes', async () => {
      async function* gen() { yield new Uint8Array([1, 2]) }
      const reader = new FramedReader(gen())
      await expect(reader.readExact(5)).rejects.toThrow('unexpected end of stream')
    })
  })
})
