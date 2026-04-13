import type { Libp2p } from 'libp2p'
import type { PeerId, Stream } from '@libp2p/interface'
import { createHash, sign as cryptoSign, verify as cryptoVerify, createPrivateKey, createPublicKey } from 'crypto'
import type { NodeIdentity } from '../main/identity'

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

export const DEFAULT_CHUNK_SIZE = 64 * 1024  // 64 KB per frame

/**
 * Split `data` into frames: [uint32 LE size][bytes] repeated, then [uint32 LE 0] sentinel.
 * Returns an array of Buffers ready to send via stream.send().
 */
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
  // End-of-stream sentinel
  const sentinel = Buffer.allocUnsafe(4)
  sentinel.writeUInt32LE(0, 0)
  frames.push(sentinel)
  return frames
}

/**
 * Reassemble frames produced by encodeChunked back into a single Buffer.
 * Stops at the first zero-size frame (end sentinel).
 */
export function decodeChunked(frames: Buffer[]): Buffer {
  const chunks: Buffer[] = []
  for (const frame of frames) {
    const size = frame.readUInt32LE(0)
    if (size === 0) break
    chunks.push(frame.slice(4, 4 + size))
  }
  return Buffer.concat(chunks)
}

// ── Ed25519 signing helpers ───────────────────────────────────────────────────

/** Compute SHA-256 of the canonical wire bytes (nTokens || nEmbd || tensor). */
function tensorHash(nTokens: number, nEmbd: number, tensorBytes: Uint8Array): Buffer {
  const header = Buffer.allocUnsafe(8)
  header.writeUInt32LE(nTokens, 0)
  header.writeUInt32LE(nEmbd, 4)
  return createHash('sha256').update(header).update(tensorBytes).digest()
}

/** Build a minimal PKCS8 DER wrapper around a 32-byte Ed25519 seed. */
function buildPkcs8Der(seed: Uint8Array): Uint8Array {
  // PKCS8 structure for Ed25519:
  // 30 2e — SEQUENCE (46 bytes)
  //   02 01 00 — INTEGER 0 (version)
  //   30 05 06 03 2b 65 70 — SEQUENCE { OID 1.3.101.112 (Ed25519) }
  //   04 22 04 20 [32 bytes seed] — OCTET STRING { OCTET STRING [seed] }
  const der = new Uint8Array(48)
  const header = [0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20]
  der.set(header, 0)
  der.set(seed, 16)
  return der
}

/** Build a minimal SPKI DER wrapper around a 32-byte Ed25519 public key. */
function buildSpkiDer(pubKey: Uint8Array): Uint8Array {
  // SPKI structure for Ed25519:
  // 30 2a — SEQUENCE (42 bytes)
  //   30 05 06 03 2b 65 70 — SEQUENCE { OID 1.3.101.112 }
  //   03 21 00 [32 bytes key] — BIT STRING (1 unused bit, then key)
  const der = new Uint8Array(44)
  const header = [0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00]
  der.set(header, 0)
  der.set(pubKey, 12)
  return der
}

/** Sign `hash` with the raw Ed25519 private key seed (32 bytes). Returns 64-byte signature. */
function signHash(hash: Buffer, privateKeySeed: Uint8Array): Buffer {
  const pkcs8Der = buildPkcs8Der(privateKeySeed)
  const keyObj = createPrivateKey({ key: Buffer.from(pkcs8Der), format: 'der', type: 'pkcs8' })
  return Buffer.from(cryptoSign(null, hash, keyObj))
}

/** Verify `signature` over `hash` using raw Ed25519 public key bytes (32 bytes). */
function verifyHash(hash: Buffer, signature: Uint8Array, publicKeyBytes: Uint8Array): boolean {
  const spkiDer = buildSpkiDer(publicKeyBytes)
  const keyObj = createPublicKey({ key: Buffer.from(spkiDer), format: 'der', type: 'spki' })
  return cryptoVerify(null, hash, keyObj, Buffer.from(signature))
}

/**
 * Encode tensor + Ed25519 signature.
 * Wire format: [base message][uint32 sigLen][sig bytes][uint32 pubkeyLen][pubkey bytes]
 */
export function encodeMessageSigned(
  data: Float32Array,
  nTokens: number,
  nEmbd: number,
  privateKey: Uint8Array,
  publicKey: Uint8Array,
): Uint8Array {
  const base = encodeMessage(data, nTokens, nEmbd)
  // Tensor bytes start at offset 8 in base
  const tensorBytes = base.slice(8)
  const hash = tensorHash(nTokens, nEmbd, tensorBytes)
  const sig = signHash(hash, privateKey)

  const sigHeader = Buffer.allocUnsafe(4)
  sigHeader.writeUInt32LE(sig.byteLength, 0)
  const pubkeyHeader = Buffer.allocUnsafe(4)
  pubkeyHeader.writeUInt32LE(publicKey.byteLength, 0)

  return Buffer.concat([base, sigHeader, sig, pubkeyHeader, Buffer.from(publicKey)])
}

/**
 * Decode a signed message. Throws if signature verification fails.
 * Returns the same shape as decodeMessage().
 */
export function decodeMessageSigned(buf: Buffer): { nTokens: number; nEmbd: number; data: Float32Array } {
  const { nTokens, nEmbd, data } = decodeMessage(buf)
  const tensorByteLen = nTokens * nEmbd * 4
  const sigOffset = 8 + tensorByteLen

  if (buf.byteLength < sigOffset + 8) {
    throw new Error('v2 message: missing signature fields')
  }

  const sigLen = buf.readUInt32LE(sigOffset)
  if (sigOffset + 4 + sigLen > buf.byteLength) {
    throw new Error('v2 message: sig bytes extend beyond buffer')
  }
  const sig = buf.slice(sigOffset + 4, sigOffset + 4 + sigLen)
  const pubkeyOffset = sigOffset + 4 + sigLen
  if (pubkeyOffset + 4 > buf.byteLength) {
    throw new Error('v2 message: pubkey length field missing')
  }
  const pubkeyLen = buf.readUInt32LE(pubkeyOffset)
  if (pubkeyOffset + 4 + pubkeyLen > buf.byteLength) {
    throw new Error('v2 message: pubkey bytes extend beyond buffer')
  }
  const pubkey = buf.slice(pubkeyOffset + 4, pubkeyOffset + 4 + pubkeyLen)

  const tensorBytes = buf.slice(8, sigOffset)
  const hash = tensorHash(nTokens, nEmbd, tensorBytes)

  if (!verifyHash(hash, sig, pubkey)) {
    throw new Error('Inference message signature verification failed — possible tensor tampering')
  }

  return { nTokens, nEmbd, data }
}

export const INFERENCE_PROTOCOL_V2 = '/opencoral/inference/2.0.0'

async function sendChunked(stream: Stream, data: Uint8Array): Promise<void> {
  for (const frame of encodeChunked(data)) {
    await sendWithBackpressure(stream, frame)
  }
}

async function collectChunkedStream(
  stream: AsyncIterable<Uint8Array | { subarray(): Uint8Array }>,
): Promise<Buffer> {
  // Collect all raw wire bytes, then re-parse as length-prefixed frames
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

export async function registerInferenceHandlerV2(
  libp2p: Libp2p,
  handler: InferenceHandler,
  identity: NodeIdentity,
): Promise<void> {
  await libp2p.handle(INFERENCE_PROTOCOL_V2, async (stream) => {
    try {
      const buf = await collectChunkedStream(stream)
      const { nTokens, nEmbd, data } = decodeMessageSigned(buf)
      const output = await handler(data, nTokens, nEmbd)
      const encoded = encodeMessageSigned(output, nTokens, nEmbd, identity.privateKey, identity.publicKey)
      await sendChunked(stream, encoded)
      await stream.close()
    } catch (err) {
      stream.abort(err instanceof Error ? err : new Error(String(err)))
    }
  }, { force: true })
}

export async function sendInferenceRequestV2(
  libp2p: Libp2p,
  peerId: PeerId,
  input: Float32Array,
  nTokens: number,
  nEmbd: number,
  identity: NodeIdentity,
): Promise<Float32Array> {
  const stream = await libp2p.dialProtocol(peerId, INFERENCE_PROTOCOL_V2)
  const request = encodeMessageSigned(input, nTokens, nEmbd, identity.privateKey, identity.publicKey)
  await sendChunked(stream, request)
  const [buf] = await Promise.all([collectChunkedStream(stream), stream.close().catch(() => {})])
  return decodeMessageSigned(buf).data
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
