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
