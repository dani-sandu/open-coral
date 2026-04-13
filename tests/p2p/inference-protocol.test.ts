import { describe, it, expect, beforeAll, afterAll } from 'bun:test'
import { createOpenCoralNode, type OpenCoralNode } from '../../src/p2p/node'
import {
  INFERENCE_PROTOCOL,
  INFERENCE_PROTOCOL_V2,
  registerInferenceHandler,
  registerInferenceHandlerV2,
  sendInferenceRequest,
  sendInferenceRequestV2,
  type InferenceHandler,
  encodeChunked,
  decodeChunked,
  encodeMessageSigned,
  decodeMessageSigned,
} from '../../src/p2p/inference-protocol'
import { loadOrCreateIdentity, type NodeIdentity } from '../../src/main/identity'
import { mkdtempSync, rmSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'

describe('inference protocol', () => {
  let nodeA: OpenCoralNode
  let nodeB: OpenCoralNode

  beforeAll(async () => {
    nodeA = await createOpenCoralNode()
    nodeB = await createOpenCoralNode()
    await nodeA.libp2p.dial(nodeB.libp2p.getMultiaddrs()[0])
  })

  afterAll(async () => {
    await nodeA.stop()
    await nodeB.stop()
  })

  it('protocol ID is correct', () => {
    expect(INFERENCE_PROTOCOL).toBe('/opencoral/inference/1.0.0')
  })

  it('handler receives correct tensor shape', async () => {
    let receivedTokens = 0
    let receivedEmbd = 0

    const handler: InferenceHandler = async (input, nTokens, nEmbd) => {
      receivedTokens = nTokens
      receivedEmbd = nEmbd
      return input  // echo
    }
    await registerInferenceHandler(nodeB.libp2p, handler)

    const input = new Float32Array(3 * 16).fill(0.5)
    await sendInferenceRequest(nodeA.libp2p, nodeB.libp2p.peerId, input, 3, 16)

    expect(receivedTokens).toBe(3)
    expect(receivedEmbd).toBe(16)
  })

  it('response is a Float32Array with the correct length', async () => {
    const handler: InferenceHandler = async (input) => input  // identity

    // Unregister any previous handler and re-register
    try { await nodeB.libp2p.unhandle(INFERENCE_PROTOCOL) } catch {}
    await registerInferenceHandler(nodeB.libp2p, handler)

    const nTokens = 2
    const nEmbd = 8
    const input = new Float32Array(nTokens * nEmbd).fill(0.1)
    const output = await sendInferenceRequest(nodeA.libp2p, nodeB.libp2p.peerId, input, nTokens, nEmbd)

    expect(output).toBeInstanceOf(Float32Array)
    expect(output.length).toBe(nTokens * nEmbd)
  })

  it('response values match echo handler output', async () => {
    try { await nodeB.libp2p.unhandle(INFERENCE_PROTOCOL) } catch {}
    await registerInferenceHandler(nodeB.libp2p, async (input) => input)

    const input = new Float32Array([1.1, 2.2, 3.3, 4.4])
    const output = await sendInferenceRequest(nodeA.libp2p, nodeB.libp2p.peerId, input, 1, 4)

    // Float32 round-trip tolerance (IEEE 754 single precision)
    for (let i = 0; i < input.length; i++) {
      expect(output[i]).toBeCloseTo(input[i], 5)
    }
  })

  it('roundtrip with multi-token batch', async () => {
    try { await nodeB.libp2p.unhandle(INFERENCE_PROTOCOL) } catch {}
    await registerInferenceHandler(nodeB.libp2p, async (input) => input)

    const nTokens = 4
    const nEmbd = 8
    const input = new Float32Array(nTokens * nEmbd).fill(0.2)
    const output = await sendInferenceRequest(nodeA.libp2p, nodeB.libp2p.peerId, input, nTokens, nEmbd)
    expect(output.length).toBe(nTokens * nEmbd)
  })
})

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

describe('inference protocol v2 — chunked framing', () => {
  let nodeE: OpenCoralNode
  let nodeF: OpenCoralNode
  let identity: NodeIdentity
  let identityDir: string

  beforeAll(async () => {
    identityDir = mkdtempSync(join(tmpdir(), 'coral-v2-test-'))
    identity = await loadOrCreateIdentity(identityDir)
    nodeE = await createOpenCoralNode()
    nodeF = await createOpenCoralNode()
    await nodeE.libp2p.dial(nodeF.libp2p.getMultiaddrs()[0])
  })

  afterAll(async () => {
    await nodeE.stop()
    await nodeF.stop()
    rmSync(identityDir, { recursive: true })
  })

  it('INFERENCE_PROTOCOL_V2 is correct string', () => {
    expect(INFERENCE_PROTOCOL_V2).toBe('/opencoral/inference/2.0.0')
  })

  it('v2 roundtrip: sends and receives correct tensor shape', async () => {
    await registerInferenceHandlerV2(nodeF.libp2p, async (input, nTokens, nEmbd) => {
      expect(nTokens).toBe(4)
      expect(nEmbd).toBe(8)
      return input
    }, identity)

    const input = new Float32Array(4 * 8).fill(0.5)
    const output = await sendInferenceRequestV2(nodeE.libp2p, nodeF.libp2p.peerId, input, 4, 8, identity)

    expect(output).toBeInstanceOf(Float32Array)
    expect(output.length).toBe(4 * 8)
    for (let i = 0; i < input.length; i++) {
      expect(output[i]).toBeCloseTo(0.5, 5)
    }
  })

  it('v2 roundtrip: second request on same connection succeeds', async () => {
    // Previously this required fresh nodes due to Bun's createCipheriv nonce issue.
    // pureJsCrypto (@noble/ciphers) handles incremented nonces correctly across all runtimes.
    await registerInferenceHandlerV2(nodeF.libp2p, async (input) => input, identity)

    const nTokens = 16
    const nEmbd = 8
    const input = new Float32Array(nTokens * nEmbd).fill(0.1)
    const output = await sendInferenceRequestV2(nodeE.libp2p, nodeF.libp2p.peerId, input, nTokens, nEmbd, identity)
    expect(output.length).toBe(nTokens * nEmbd)
  })
})

describe('signed tensor messages', () => {
  it('encodeMessageSigned / decodeMessageSigned roundtrip', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'coral-id-test-'))
    try {
      const identity = await loadOrCreateIdentity(dir)
      const data = new Float32Array([1.1, 2.2, 3.3, 4.4])
      const buf = Buffer.from(encodeMessageSigned(data, 1, 4, identity.privateKey, identity.publicKey))
      const { nTokens, nEmbd, data: decoded } = decodeMessageSigned(buf)
      expect(nTokens).toBe(1)
      expect(nEmbd).toBe(4)
      for (let i = 0; i < data.length; i++) expect(decoded[i]).toBeCloseTo(data[i], 5)
    } finally {
      rmSync(dir, { recursive: true })
    }
  })

  it('decodeMessageSigned throws on tampered tensor', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'coral-id-test2-'))
    try {
      const identity = await loadOrCreateIdentity(dir)
      const data = new Float32Array([1, 2, 3, 4])
      const buf = Buffer.from(encodeMessageSigned(data, 1, 4, identity.privateKey, identity.publicKey))
      // Tamper with tensor byte at offset 8
      buf[8] ^= 0xFF
      expect(() => decodeMessageSigned(buf)).toThrow('signature verification failed')
    } finally {
      rmSync(dir, { recursive: true })
    }
  })
})
