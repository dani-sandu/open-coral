import { describe, it, expect, beforeAll, afterAll } from 'bun:test'
import { createOpenCoralNode, type OpenCoralNode } from '../../src/p2p/node'
import {
  INFERENCE_PROTOCOL_V3,
  registerInferenceHandlerV3,
  sendInferenceRequestV3,
  encodeChunked,
  decodeChunked,
  encodeMessageV3,
  decodeMessageV3,
} from '../../src/p2p/inference-protocol'
import { loadOrCreateIdentity, type NodeIdentity } from '../../src/main/identity'
import { mkdtempSync, rmSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'

describe('chunked framing helpers', () => {
  it('encodeChunked produces correct frame structure', () => {
    const data = new Uint8Array([1, 2, 3, 4, 5])
    const chunkSize = 3
    const frames = encodeChunked(data, chunkSize)
    // Frame 0: [uint32(3)][1,2,3]
    // Frame 1: [uint32(2)][4,5]
    // Frame 2: [uint32(0)]  ← end sentinel
    expect(frames).toHaveLength(3)
    expect(frames[0].readUInt32LE(0)).toBe(3)
    expect(Array.from(frames[0].slice(4))).toEqual([1, 2, 3])
    expect(frames[1].readUInt32LE(0)).toBe(2)
    expect(Array.from(frames[1].slice(4))).toEqual([4, 5])
    expect(frames[2].readUInt32LE(0)).toBe(0)
    expect(frames[2].byteLength).toBe(4)
  })

  it('decodeChunked reassembles frames into original data', () => {
    const original = Buffer.from([10, 20, 30, 40, 50, 60])
    const frames = encodeChunked(original, 4)
    const result = decodeChunked(frames)
    expect(Array.from(result)).toEqual([10, 20, 30, 40, 50, 60])
  })

  it('decodeChunked handles data that fits in one chunk', () => {
    const data = Buffer.from([7, 8, 9])
    const frames = encodeChunked(data, 1024)
    const result = decodeChunked(frames)
    expect(Array.from(result)).toEqual([7, 8, 9])
  })

  it('decodeChunked returns empty buffer for zero-byte input', () => {
    const frames = encodeChunked(Buffer.alloc(0), 64)
    const result = decodeChunked(frames)
    expect(result.byteLength).toBe(0)
  })
})


describe('inference protocol v3 — requestId', () => {
  let identity: NodeIdentity
  let identityDir: string

  beforeAll(async () => {
    identityDir = mkdtempSync(join(tmpdir(), 'coral-v3-test-'))
    identity = await loadOrCreateIdentity(identityDir)
  })

  afterAll(() => {
    rmSync(identityDir, { recursive: true })
  })

  it('INFERENCE_PROTOCOL_V3 is correct string', () => {
    expect(INFERENCE_PROTOCOL_V3).toBe('/opencoral/inference/3.0.0')
  })

  it('encodeMessageV3 / decodeMessageV3 roundtrip preserves requestId', () => {
    const data = new Float32Array([1.1, 2.2, 3.3, 4.4])
    const requestId = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
    const buf = Buffer.from(encodeMessageV3(data, 1, 4, requestId, identity.privateKey, identity.publicKey))
    const decoded = decodeMessageV3(buf)
    expect(decoded.requestId).toBe(requestId)
    expect(decoded.nTokens).toBe(1)
    expect(decoded.nEmbd).toBe(4)
    for (let i = 0; i < data.length; i++) {
      expect(decoded.data[i]).toBeCloseTo(data[i], 5)
    }
  })

  it('decodeMessageV3 throws on tampered requestId', () => {
    const data = new Float32Array([1, 2, 3, 4])
    const requestId = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
    const buf = Buffer.from(encodeMessageV3(data, 1, 4, requestId, identity.privateKey, identity.publicKey))
    buf[3] ^= 0xFF
    expect(() => decodeMessageV3(buf)).toThrow('signature verification failed')
  })

  it('version byte is 3', () => {
    const data = new Float32Array([1, 2])
    const buf = Buffer.from(encodeMessageV3(data, 1, 2, 'test-id', identity.privateKey, identity.publicKey))
    expect(buf[0]).toBe(3)
  })
})

describe('inference protocol v3 — network roundtrip', () => {
  let nodeG: OpenCoralNode
  let nodeH: OpenCoralNode
  let identity: NodeIdentity
  let identityDir: string

  beforeAll(async () => {
    identityDir = mkdtempSync(join(tmpdir(), 'coral-v3-net-'))
    identity = await loadOrCreateIdentity(identityDir)
    nodeG = await createOpenCoralNode()
    nodeH = await createOpenCoralNode()
    await nodeG.libp2p.dial(nodeH.libp2p.getMultiaddrs()[0])
  })

  afterAll(async () => {
    await nodeG.stop()
    await nodeH.stop()
    rmSync(identityDir, { recursive: true })
  })

  it('v3 roundtrip preserves requestId and tensor data', async () => {
    await registerInferenceHandlerV3(nodeH.libp2p, async (input) => input, identity)

    const input = new Float32Array(4 * 8).fill(0.3)
    const requestId = 'test-req-001'
    const output = await sendInferenceRequestV3(
      nodeG.libp2p, nodeH.libp2p.peerId, input, 4, 8, requestId, identity,
    )

    expect(output).toBeInstanceOf(Float32Array)
    expect(output.length).toBe(4 * 8)
    for (let i = 0; i < input.length; i++) {
      expect(output[i]).toBeCloseTo(0.3, 5)
    }
  })
})

describe('inference protocol v4 — float16 wire format', () => {
  let v4identity: import('../../src/main/identity').NodeIdentity
  let v4dir: string
  beforeAll(async () => {
    const { mkdtempSync } = await import('fs')
    const { tmpdir } = await import('os')
    const { join } = await import('path')
    const { loadOrCreateIdentity } = await import('../../src/main/identity')
    v4dir = mkdtempSync(join(tmpdir(), 'coral-v4-'))
    v4identity = await loadOrCreateIdentity(v4dir)
  })
  afterAll(async () => {
    const { rmSync } = await import('fs')
    rmSync(v4dir, { recursive: true })
  })

  it('INFERENCE_PROTOCOL_V4 is the 4.0.0 string', async () => {
    const { INFERENCE_PROTOCOL_V4 } = await import('../../src/p2p/inference-protocol')
    expect(INFERENCE_PROTOCOL_V4).toBe('/opencoral/inference/4.0.0')
  })

  it('V4 float16 round-trips within tolerance and sets version byte 4', async () => {
    const { encodeMessageV4, decodeMessageV4 } = await import('../../src/p2p/inference-protocol')
    const data = new Float32Array([0.1, -0.25, 0.5, 1.0, -1.0, 0.0, 3.14, -2.5])
    const buf = Buffer.from(encodeMessageV4(data, 1, 8, 'req-v4', v4identity.privateKey, v4identity.publicKey, 'float16'))
    expect(buf[0]).toBe(4)
    const decoded = decodeMessageV4(buf)
    expect(decoded.requestId).toBe('req-v4')
    expect(decoded.nTokens).toBe(1)
    expect(decoded.nEmbd).toBe(8)
    expect(decoded.dtype).toBe('float16')
    for (let i = 0; i < data.length; i++) {
      expect(Math.abs(decoded.data[i] - data[i])).toBeLessThanOrEqual(Math.max(Math.abs(data[i]) * 0.01, 1e-3))
    }
  })

  it('V4 float16 tensor is ~half the wire size of V3 (float32)', async () => {
    const { encodeMessageV3, encodeMessageV4 } = await import('../../src/p2p/inference-protocol')
    const n = 256
    const data = new Float32Array(n).fill(0.3)
    const v3 = encodeMessageV3(data, 1, n, 'r', v4identity.privateKey, v4identity.publicKey)
    const v4 = encodeMessageV4(data, 1, n, 'r', v4identity.privateKey, v4identity.publicKey, 'float16')
    expect(v3.byteLength - v4.byteLength).toBeGreaterThanOrEqual(n * 2 - 8)
  })

  it('V4 float32 round-trips exactly (fallback dtype)', async () => {
    const { encodeMessageV4, decodeMessageV4 } = await import('../../src/p2p/inference-protocol')
    const data = new Float32Array([0.123456, -7.89, 1000.5])
    const buf = Buffer.from(encodeMessageV4(data, 1, 3, 'f32', v4identity.privateKey, v4identity.publicKey, 'float32'))
    const decoded = decodeMessageV4(buf)
    expect(decoded.dtype).toBe('float32')
    for (let i = 0; i < data.length; i++) expect(decoded.data[i]).toBeCloseTo(data[i], 5)
  })

  it('V4 rejects a tampered tensor (signature verification)', async () => {
    const { encodeMessageV4, decodeMessageV4 } = await import('../../src/p2p/inference-protocol')
    const data = new Float32Array([1, 2, 3, 4])
    const buf = Buffer.from(encodeMessageV4(data, 1, 4, 'tamper', v4identity.privateKey, v4identity.publicKey, 'float16'))
    buf[buf.length - 40] ^= 0xff
    expect(() => decodeMessageV4(buf)).toThrow('signature verification failed')
  })
})
