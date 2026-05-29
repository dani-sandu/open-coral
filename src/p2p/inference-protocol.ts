import type { Libp2p } from 'libp2p'
import type { PeerId } from '@libp2p/interface'
import { createHash } from 'crypto'
import type { NodeIdentity } from '../main/identity'
import { encodeChunked, decodeChunked, collectChunkedStream, sendChunked, DEFAULT_CHUNK_SIZE } from './stream-utils'
import { signHash, verifyHash } from './ed25519-helpers'
import { float32ArrayToFloat16, float16ArrayToFloat32 } from './float16'

export { encodeChunked, decodeChunked, DEFAULT_CHUNK_SIZE } from './stream-utils'

export type InferenceHandler = (
  input: Float32Array,
  nTokens: number,
  nEmbd: number,
) => Promise<Float32Array>

/**
 * Initiator-observable timing breakdown of one remote inference hop, in milliseconds.
 * `waitMs` lumps network-receive and remote compute together — they cannot be
 * separated without the remote reporting its own compute time (deferred follow-up).
 */
export interface RemoteHopPhases {
  signMs: number    // encodeMessageV3: build payload + SHA-256 + ed25519 sign
  sendMs: number    // sendChunked: frame and write the request to the stream
  waitMs: number    // await the response: network receive + remote compute
  verifyMs: number  // decodeMessageV3: SHA-256 + ed25519 verify + decode
}

export const INFERENCE_PROTOCOL_V3 = '/opencoral/inference/3.0.0'

export const INFERENCE_PROTOCOL_V4 = '/opencoral/inference/4.0.0'

export type TensorDType = 'float32' | 'float16'

/**
 * Compute SHA-256 of an arbitrary payload buffer.
 * Used by V3 to hash the full wire payload (version + requestId + nTokens + nEmbd + tensor).
 */
function payloadHash(payload: Buffer): Buffer {
  return createHash('sha256').update(payload).digest()
}

/**
 * Encode a V3 inference message with a requestId field.
 *
 * Wire format:
 *   [uint8: version=3]
 *   [uint16 LE: requestId length][requestId bytes (UTF-8)]
 *   [uint32 LE: nTokens][uint32 LE: nEmbd][float32[] tensor data]
 *   [uint32 LE: sigLen][sig bytes]
 *   [uint32 LE: pubkeyLen][pubkey bytes]
 *
 * The signature covers everything from the version byte through the tensor data.
 */
export function encodeMessageV3(
  data: Float32Array,
  nTokens: number,
  nEmbd: number,
  requestId: string,
  privateKey: Uint8Array,
  publicKey: Uint8Array,
): Uint8Array {
  const requestIdBytes = Buffer.from(requestId, 'utf8')
  const tensorBytes = Buffer.from(data.buffer, data.byteOffset, data.byteLength)

  // Build the payload that will be signed: version + requestId fields + nTokens + nEmbd + tensor
  const payloadLen = 1 + 2 + requestIdBytes.byteLength + 4 + 4 + tensorBytes.byteLength
  const payload = Buffer.allocUnsafe(payloadLen)

  let offset = 0
  payload.writeUInt8(3, offset); offset += 1
  payload.writeUInt16LE(requestIdBytes.byteLength, offset); offset += 2
  requestIdBytes.copy(payload, offset); offset += requestIdBytes.byteLength
  payload.writeUInt32LE(nTokens, offset); offset += 4
  payload.writeUInt32LE(nEmbd, offset); offset += 4
  tensorBytes.copy(payload, offset)

  const hash = payloadHash(payload)
  const sig = signHash(hash, privateKey)

  const sigHeader = Buffer.allocUnsafe(4)
  sigHeader.writeUInt32LE(sig.byteLength, 0)
  const pubkeyHeader = Buffer.allocUnsafe(4)
  pubkeyHeader.writeUInt32LE(publicKey.byteLength, 0)

  return Buffer.concat([payload, sigHeader, sig, pubkeyHeader, Buffer.from(publicKey)])
}

/**
 * Decode a V3 inference message. Throws if signature verification fails.
 * Returns { requestId, nTokens, nEmbd, data }.
 */
export function decodeMessageV3(buf: Buffer): { requestId: string; nTokens: number; nEmbd: number; data: Float32Array } {
  if (buf.byteLength < 1) {
    throw new Error('v3 message: buffer too short for version byte')
  }

  const version = buf.readUInt8(0)
  if (version !== 3) {
    throw new Error(`v3 message: expected version 3, got ${version}`)
  }

  if (buf.byteLength < 3) {
    throw new Error('v3 message: buffer too short for requestId length field')
  }
  const requestIdLen = buf.readUInt16LE(1)

  let offset = 3
  if (buf.byteLength < offset + requestIdLen) {
    throw new Error('v3 message: requestId bytes extend beyond buffer')
  }
  const requestId = buf.slice(offset, offset + requestIdLen).toString('utf8')
  offset += requestIdLen

  if (buf.byteLength < offset + 8) {
    throw new Error('v3 message: buffer too short for nTokens/nEmbd fields')
  }
  const nTokens = buf.readUInt32LE(offset); offset += 4
  const nEmbd = buf.readUInt32LE(offset); offset += 4

  const tensorByteLen = nTokens * nEmbd * 4
  if (buf.byteLength < offset + tensorByteLen) {
    throw new Error(`v3 message: tensor bytes extend beyond buffer`)
  }
  const tensorEnd = offset + tensorByteLen

  // The payload that was signed is everything before the signature fields
  const payload = buf.slice(0, tensorEnd)
  const hash = payloadHash(payload)

  offset = tensorEnd
  if (buf.byteLength < offset + 4) {
    throw new Error('v3 message: missing signature length field')
  }
  const sigLen = buf.readUInt32LE(offset); offset += 4
  if (buf.byteLength < offset + sigLen) {
    throw new Error('v3 message: sig bytes extend beyond buffer')
  }
  const sig = buf.slice(offset, offset + sigLen); offset += sigLen

  if (buf.byteLength < offset + 4) {
    throw new Error('v3 message: missing pubkey length field')
  }
  const pubkeyLen = buf.readUInt32LE(offset); offset += 4
  if (buf.byteLength < offset + pubkeyLen) {
    throw new Error('v3 message: pubkey bytes extend beyond buffer')
  }
  const pubkey = buf.slice(offset, offset + pubkeyLen)

  if (!verifyHash(hash, sig, pubkey)) {
    throw new Error('V3 inference message signature verification failed — possible tensor tampering')
  }

  const floatBuf = Buffer.allocUnsafe(tensorByteLen)
  buf.copy(floatBuf, 0, tensorEnd - tensorByteLen, tensorEnd)
  const data = new Float32Array(floatBuf.buffer)

  return { requestId, nTokens, nEmbd, data }
}

/**
 * V4 wire format = V3 + a 1-byte dtype field before the tensor.
 *   [uint8 version=4][uint16 reqIdLen][reqId][uint32 nTokens][uint32 nEmbd]
 *   [uint8 dtype: 0=float32,1=float16][tensor][uint32 sigLen][sig][uint32 pubLen][pub]
 * The signature covers version..tensor (same scheme as V3).
 */
export function encodeMessageV4(
  data: Float32Array,
  nTokens: number,
  nEmbd: number,
  requestId: string,
  privateKey: Uint8Array,
  publicKey: Uint8Array,
  dtype: TensorDType = 'float16',
): Uint8Array {
  const requestIdBytes = Buffer.from(requestId, 'utf8')
  const tensorBytes = dtype === 'float16'
    ? Buffer.from(float32ArrayToFloat16(data).buffer)
    : Buffer.from(data.buffer, data.byteOffset, data.byteLength)
  const dtypeByte = dtype === 'float16' ? 1 : 0

  const payloadLen = 1 + 2 + requestIdBytes.byteLength + 4 + 4 + 1 + tensorBytes.byteLength
  const payload = Buffer.allocUnsafe(payloadLen)

  let offset = 0
  payload.writeUInt8(4, offset); offset += 1
  payload.writeUInt16LE(requestIdBytes.byteLength, offset); offset += 2
  requestIdBytes.copy(payload, offset); offset += requestIdBytes.byteLength
  payload.writeUInt32LE(nTokens, offset); offset += 4
  payload.writeUInt32LE(nEmbd, offset); offset += 4
  payload.writeUInt8(dtypeByte, offset); offset += 1
  tensorBytes.copy(payload, offset)

  const hash = payloadHash(payload)
  const sig = signHash(hash, privateKey)

  const sigHeader = Buffer.allocUnsafe(4)
  sigHeader.writeUInt32LE(sig.byteLength, 0)
  const pubkeyHeader = Buffer.allocUnsafe(4)
  pubkeyHeader.writeUInt32LE(publicKey.byteLength, 0)

  return Buffer.concat([payload, sigHeader, sig, pubkeyHeader, Buffer.from(publicKey)])
}

export function decodeMessageV4(buf: Buffer): {
  requestId: string; nTokens: number; nEmbd: number; dtype: TensorDType; data: Float32Array
} {
  if (buf.byteLength < 1) throw new Error('v4 message: buffer too short for version byte')
  const version = buf.readUInt8(0)
  if (version !== 4) throw new Error(`v4 message: expected version 4, got ${version}`)
  if (buf.byteLength < 3) throw new Error('v4 message: buffer too short for requestId length field')

  const requestIdLen = buf.readUInt16LE(1)
  let offset = 3
  if (buf.byteLength < offset + requestIdLen) throw new Error('v4 message: requestId bytes extend beyond buffer')
  const requestId = buf.slice(offset, offset + requestIdLen).toString('utf8')
  offset += requestIdLen

  if (buf.byteLength < offset + 9) throw new Error('v4 message: buffer too short for nTokens/nEmbd/dtype fields')
  const nTokens = buf.readUInt32LE(offset); offset += 4
  const nEmbd = buf.readUInt32LE(offset); offset += 4
  const dtypeByte = buf.readUInt8(offset); offset += 1
  const dtype: TensorDType = dtypeByte === 1 ? 'float16' : 'float32'

  const bytesPerElem = dtype === 'float16' ? 2 : 4
  const tensorByteLen = nTokens * nEmbd * bytesPerElem
  if (buf.byteLength < offset + tensorByteLen) throw new Error('v4 message: tensor bytes extend beyond buffer')
  const tensorEnd = offset + tensorByteLen

  const payload = buf.slice(0, tensorEnd)
  const hash = payloadHash(payload)

  offset = tensorEnd
  if (buf.byteLength < offset + 4) throw new Error('v4 message: missing signature length field')
  const sigLen = buf.readUInt32LE(offset); offset += 4
  if (buf.byteLength < offset + sigLen) throw new Error('v4 message: sig bytes extend beyond buffer')
  const sig = buf.slice(offset, offset + sigLen); offset += sigLen

  if (buf.byteLength < offset + 4) throw new Error('v4 message: missing pubkey length field')
  const pubkeyLen = buf.readUInt32LE(offset); offset += 4
  if (buf.byteLength < offset + pubkeyLen) throw new Error('v4 message: pubkey bytes extend beyond buffer')
  const pubkey = buf.slice(offset, offset + pubkeyLen)

  if (!verifyHash(hash, sig, pubkey)) {
    throw new Error('V4 inference message signature verification failed — possible tensor tampering')
  }

  let data: Float32Array
  if (dtype === 'float16') {
    const u16 = new Uint16Array(nTokens * nEmbd)
    for (let i = 0; i < u16.length; i++) u16[i] = buf.readUInt16LE(tensorEnd - tensorByteLen + i * 2)
    data = float16ArrayToFloat32(u16)
  } else {
    const floatBuf = Buffer.allocUnsafe(tensorByteLen)
    buf.copy(floatBuf, 0, tensorEnd - tensorByteLen, tensorEnd)
    data = new Float32Array(floatBuf.buffer)
  }

  return { requestId, nTokens, nEmbd, dtype, data }
}

export async function registerInferenceHandlerV3(
  libp2p: Libp2p,
  handler: InferenceHandler,
  identity: NodeIdentity,
): Promise<void> {
  await libp2p.handle(INFERENCE_PROTOCOL_V3, async (stream) => {
    try {
      const buf = await collectChunkedStream(stream)
      const { requestId, nTokens, nEmbd, data } = decodeMessageV3(buf)
      const output = await handler(data, nTokens, nEmbd)
      const encoded = encodeMessageV3(output, nTokens, nEmbd, requestId, identity.privateKey, identity.publicKey)
      await sendChunked(stream, encoded)
      await stream.close()
    } catch (err) {
      stream.abort(err instanceof Error ? err : new Error(String(err)))
    }
  }, { force: true })
}

export async function sendInferenceRequestV3(
  libp2p: Libp2p,
  peerId: PeerId,
  input: Float32Array,
  nTokens: number,
  nEmbd: number,
  requestId: string,
  identity: NodeIdentity,
  onPhases?: (phases: RemoteHopPhases) => void,
): Promise<Float32Array> {
  const stream = await libp2p.dialProtocol(peerId, INFERENCE_PROTOCOL_V3)

  const tSignStart = performance.now()
  const request = encodeMessageV3(input, nTokens, nEmbd, requestId, identity.privateKey, identity.publicKey)
  const tSendStart = performance.now()
  await sendChunked(stream, request)
  const tWaitStart = performance.now()
  const [buf] = await Promise.all([collectChunkedStream(stream), stream.close().catch(() => {})])
  const tVerifyStart = performance.now()
  const decoded = decodeMessageV3(buf)
  const tEnd = performance.now()

  onPhases?.({
    signMs: tSendStart - tSignStart,
    sendMs: tWaitStart - tSendStart,
    waitMs: tVerifyStart - tWaitStart,
    verifyMs: tEnd - tVerifyStart,
  })
  return decoded.data
}

export async function registerInferenceHandlerV4(
  libp2p: Libp2p,
  handler: InferenceHandler,
  identity: NodeIdentity,
): Promise<void> {
  await libp2p.handle(INFERENCE_PROTOCOL_V4, async (stream) => {
    try {
      const buf = await collectChunkedStream(stream)
      const { requestId, nTokens, nEmbd, data } = decodeMessageV4(buf)
      const output = await handler(data, nTokens, nEmbd)
      const encoded = encodeMessageV4(output, nTokens, nEmbd, requestId, identity.privateKey, identity.publicKey, 'float16')
      await sendChunked(stream, encoded)
      await stream.close()
    } catch (err) {
      stream.abort(err instanceof Error ? err : new Error(String(err)))
    }
  }, { force: true })
}

export interface NegotiatedSendOptions {
  onPhases?: (phases: RemoteHopPhases) => void
  onWire?: (info: { protocol: string; requestBytes: number }) => void
}

/**
 * Send an inference request, preferring V4 (fp16) and falling back to V3 (fp32)
 * based on what the peer supports (libp2p multistream-select over [V4, V3]).
 */
export async function sendInferenceRequestNegotiated(
  libp2p: Libp2p,
  peerId: PeerId,
  input: Float32Array,
  nTokens: number,
  nEmbd: number,
  requestId: string,
  identity: NodeIdentity,
  opts: NegotiatedSendOptions = {},
): Promise<Float32Array> {
  const stream = await (libp2p as any).dialProtocol(peerId, [INFERENCE_PROTOCOL_V4, INFERENCE_PROTOCOL_V3])
  const negotiated = (stream as { protocol?: string }).protocol ?? INFERENCE_PROTOCOL_V3
  const useV4 = negotiated === INFERENCE_PROTOCOL_V4

  const tSignStart = performance.now()
  const request = useV4
    ? encodeMessageV4(input, nTokens, nEmbd, requestId, identity.privateKey, identity.publicKey, 'float16')
    : encodeMessageV3(input, nTokens, nEmbd, requestId, identity.privateKey, identity.publicKey)
  const tSendStart = performance.now()
  await sendChunked(stream, request)
  const tWaitStart = performance.now()
  const [buf] = await Promise.all([collectChunkedStream(stream), stream.close().catch(() => {})])
  const tVerifyStart = performance.now()
  const data = useV4 ? decodeMessageV4(buf).data : decodeMessageV3(buf).data
  const tEnd = performance.now()

  opts.onPhases?.({
    signMs: tSendStart - tSignStart,
    sendMs: tWaitStart - tSendStart,
    waitMs: tVerifyStart - tWaitStart,
    verifyMs: tEnd - tVerifyStart,
  })
  opts.onWire?.({ protocol: negotiated, requestBytes: request.byteLength })
  return data
}


