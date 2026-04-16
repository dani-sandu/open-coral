import { createPrivateKey, createPublicKey, sign as cryptoSign, verify as cryptoVerify } from 'crypto'

/** Build a minimal PKCS8 DER wrapper around a 32-byte Ed25519 seed. */
export function buildPkcs8Der(seed: Uint8Array): Uint8Array {
  const der = new Uint8Array(48)
  const header = [0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20]
  der.set(header, 0)
  der.set(seed, 16)
  return der
}

/** Build a minimal SPKI DER wrapper around a 32-byte Ed25519 public key. */
export function buildSpkiDer(pubKey: Uint8Array): Uint8Array {
  const der = new Uint8Array(44)
  const header = [0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00]
  der.set(header, 0)
  der.set(pubKey, 12)
  return der
}

/** Sign `hash` with raw Ed25519 private key seed (32 bytes). Returns 64-byte signature. */
export function signHash(hash: Buffer, privateKeySeed: Uint8Array): Buffer {
  const pkcs8Der = buildPkcs8Der(privateKeySeed)
  const keyObj = createPrivateKey({ key: Buffer.from(pkcs8Der), format: 'der', type: 'pkcs8' })
  return Buffer.from(cryptoSign(null, hash, keyObj))
}

/** Verify `signature` over `hash` using raw Ed25519 public key bytes (32 bytes). */
export function verifyHash(hash: Buffer, signature: Uint8Array, publicKeyBytes: Uint8Array): boolean {
  const spkiDer = buildSpkiDer(publicKeyBytes)
  const keyObj = createPublicKey({ key: Buffer.from(spkiDer), format: 'der', type: 'spki' })
  return cryptoVerify(null, hash, keyObj, Buffer.from(signature))
}
