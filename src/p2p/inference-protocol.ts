import type { Libp2p } from 'libp2p'
import type { PeerId } from '@libp2p/interface'
import { createHash } from 'crypto'
import type { NodeIdentity } from '../main/identity'
import { encodeChunked, decodeChunked, collectChunkedStream, sendChunked, DEFAULT_CHUNK_SIZE } from './stream-utils'
import { signHash, verifyHash } from './ed25519-helpers'

export { encodeChunked, decodeChunked, DEFAULT_CHUNK_SIZE } from './stream-utils'

export type InferenceHandler = (
  input: Float32Array,
  nTokens: number,
  nEmbd: number,
) => Promise<Float32Array>

export const INFERENCE_PROTOCOL_V3 = '/opencoral/inference/3.0.0'

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
): Promise<Float32Array> {
  const stream = await libp2p.dialProtocol(peerId, INFERENCE_PROTOCOL_V3)
  const request = encodeMessageV3(input, nTokens, nEmbd, requestId, identity.privateKey, identity.publicKey)
  await sendChunked(stream, request)
  const [buf] = await Promise.all([collectChunkedStream(stream), stream.close().catch(() => {})])
  return decodeMessageV3(buf).data
}


