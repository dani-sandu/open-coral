// TODO: Integrate with inference-protocol.ts — decodeMessageV3 should verify
// the sender's public key against TrustedKeyStore before accepting tensors.
// See spec Section 2.4 "Integration with inference protocol" for the required
// flow: extract key from wire → lookup in store → reject on mismatch → lazy
// query bootstrap if unknown. Graceful degradation: if bootstrap unreachable,
// fall back to signature-only verification with a warning log.
import type { Libp2p } from 'libp2p'
import type { PeerId } from '@libp2p/interface'
import { createHash } from 'crypto'
import type { NodeIdentity } from '../main/identity'
import { collectStream, sendWithBackpressure } from './stream-utils'
import { signHash, verifyHash } from './ed25519-helpers'

export const KEY_REGISTRY_PROTOCOL = '/opencoral/keyregistry/1.0.0'

const MSG_REGISTER = 0x01
const MSG_LOOKUP   = 0x02
const STATUS_OK    = 0x00
const STATUS_ERR   = 0x01

/** Compute SHA-256 over `peerId + publicKeyHex`. */
function registrationHash(peerId: string, publicKeyHex: string): Buffer {
  return createHash('sha256').update(peerId).update(publicKeyHex).digest()
}

// ── TrustedKeyStore ───────────────────────────────────────────────────────────

export class TrustedKeyStore {
  private readonly map = new Map<string, Uint8Array>()

  getKey(peerId: string): Uint8Array | null {
    return this.map.get(peerId) ?? null
  }

  setKey(peerId: string, publicKey: Uint8Array): void {
    this.map.set(peerId, publicKey)
  }

  hasKey(peerId: string): boolean {
    return this.map.has(peerId)
  }
}

// ── Bootstrap-side handler ────────────────────────────────────────────────────

export async function registerKeyRegistryHandler(libp2p: Libp2p, store: TrustedKeyStore): Promise<void> {
  await libp2p.handle(KEY_REGISTRY_PROTOCOL, async (stream) => {
    try {
      const raw = await collectStream(stream)
      if (raw.byteLength < 1) {
        await sendWithBackpressure(stream, Buffer.from([STATUS_ERR]))
        await stream.close()
        return
      }

      const msgType = raw[0]
      const body = raw.slice(1)

      if (msgType === MSG_REGISTER) {
        // Body is JSON { peerId, publicKey (hex), signature (hex) }
        let parsed: { peerId: string; publicKey: string; signature: string }
        try {
          parsed = JSON.parse(body.toString('utf-8'))
        } catch {
          await sendWithBackpressure(stream, Buffer.from([STATUS_ERR, ...Buffer.from('invalid JSON', 'utf-8')]))
          await stream.close()
          return
        }

        const { peerId, publicKey: publicKeyHex, signature: signatureHex } = parsed

        // Verify signature
        const hash = registrationHash(peerId, publicKeyHex)
        const publicKeyBytes = Buffer.from(publicKeyHex, 'hex')
        const signatureBytes = Buffer.from(signatureHex, 'hex')
        const valid = verifyHash(hash, signatureBytes, publicKeyBytes)

        if (!valid) {
          const errMsg = Buffer.from('invalid signature', 'utf-8')
          const resp = Buffer.concat([Buffer.from([STATUS_ERR]), errMsg])
          await sendWithBackpressure(stream, resp)
          await stream.close()
          return
        }

        // Check for conflicting registration
        if (store.hasKey(peerId)) {
          const existing = store.getKey(peerId)!
          const incomingHex = Buffer.from(publicKeyBytes).toString('hex')
          const existingHex = Buffer.from(existing).toString('hex')
          if (incomingHex !== existingHex) {
            const errMsg = Buffer.from('key already registered with different value', 'utf-8')
            const resp = Buffer.concat([Buffer.from([STATUS_ERR]), errMsg])
            await sendWithBackpressure(stream, resp)
            await stream.close()
            return
          }
        }

        store.setKey(peerId, new Uint8Array(publicKeyBytes))
        await sendWithBackpressure(stream, Buffer.from([STATUS_OK]))
        await stream.close()

      } else if (msgType === MSG_LOOKUP) {
        const targetPeerId = body.toString('utf-8')
        const key = store.getKey(targetPeerId)

        if (key === null) {
          const errMsg = Buffer.from('not found', 'utf-8')
          const resp = Buffer.concat([Buffer.from([STATUS_ERR]), errMsg])
          await sendWithBackpressure(stream, resp)
        } else {
          const resp = Buffer.concat([Buffer.from([STATUS_OK]), Buffer.from(key)])
          await sendWithBackpressure(stream, resp)
        }
        await stream.close()

      } else {
        const errMsg = Buffer.from(`unknown message type: ${msgType}`, 'utf-8')
        const resp = Buffer.concat([Buffer.from([STATUS_ERR]), errMsg])
        await sendWithBackpressure(stream, resp)
        await stream.close()
      }
    } catch (err) {
      stream.abort(err instanceof Error ? err : new Error(String(err)))
    }
  }, { force: true })
}

// ── Client: register with bootstrap ─────────────────────────────────────────

export async function registerWithBootstrap(
  libp2p: Libp2p,
  bootstrapPeerId: PeerId,
  identity: NodeIdentity,
): Promise<void> {
  const peerId = libp2p.peerId.toString()
  const publicKeyHex = Buffer.from(identity.publicKey).toString('hex')

  const hash = registrationHash(peerId, publicKeyHex)
  const sig = signHash(hash, identity.privateKey)
  const signatureHex = sig.toString('hex')

  const json = JSON.stringify({ peerId, publicKey: publicKeyHex, signature: signatureHex })
  const jsonBytes = Buffer.from(json, 'utf-8')

  // Build request: [MSG_REGISTER][JSON bytes]
  const request = Buffer.concat([Buffer.from([MSG_REGISTER]), jsonBytes])

  const stream = await libp2p.dialProtocol(bootstrapPeerId, KEY_REGISTRY_PROTOCOL)
  await sendWithBackpressure(stream, request)
  const [resp] = await Promise.all([collectStream(stream), stream.close().catch(() => {})])
  if (resp.byteLength < 1 || resp[0] !== STATUS_OK) {
    const errMsg = resp.byteLength > 1 ? resp.slice(1).toString('utf-8') : 'registration failed'
    throw new Error(`Key registration rejected: ${errMsg}`)
  }
}

// ── Client: lookup peer key from bootstrap ────────────────────────────────────

export async function lookupPeerKey(
  libp2p: Libp2p,
  bootstrapPeerId: PeerId,
  targetPeerId: string,
): Promise<Uint8Array | null> {
  const targetBytes = Buffer.from(targetPeerId, 'utf-8')

  // Build request: [MSG_LOOKUP][target peerId bytes]
  const request = Buffer.concat([Buffer.from([MSG_LOOKUP]), targetBytes])

  const stream = await libp2p.dialProtocol(bootstrapPeerId, KEY_REGISTRY_PROTOCOL)
  await sendWithBackpressure(stream, request)
  const [resp] = await Promise.all([collectStream(stream), stream.close().catch(() => {})])
  if (resp.byteLength < 1) return null

  const status = resp[0]
  if (status !== STATUS_OK) return null

  const keyBytes = resp.slice(1)
  if (keyBytes.byteLength === 0) return null

  return new Uint8Array(keyBytes)
}
