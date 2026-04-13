import { generateKeyPairSync } from 'crypto'
import { readFileSync, writeFileSync, existsSync } from 'fs'
import { join } from 'path'

export interface NodeIdentity {
  /** Raw private key seed bytes (Ed25519, 32 bytes) */
  privateKey: Uint8Array
  /** Raw public key bytes (Ed25519, 32 bytes) */
  publicKey: Uint8Array
}

const IDENTITY_FILE = 'coral-identity.json'

/**
 * Load the node's Ed25519 identity from `dir/coral-identity.json`, or generate
 * and save a new one if the file does not exist.
 */
export async function loadOrCreateIdentity(dir: string): Promise<NodeIdentity> {
  const filePath = join(dir, IDENTITY_FILE)

  if (existsSync(filePath)) {
    try {
      const stored = JSON.parse(readFileSync(filePath, 'utf-8'))
      if (stored.privateKey && stored.publicKey) {
        return {
          privateKey: Buffer.from(stored.privateKey, 'hex'),
          publicKey:  Buffer.from(stored.publicKey,  'hex'),
        }
      }
    } catch {
      // Corrupt or incomplete file — fall through to generate a new identity
    }
  }

  // Generate new Ed25519 key pair
  const { privateKey: privKeyObj, publicKey: pubKeyObj } = generateKeyPairSync('ed25519')

  // Export raw bytes
  const privDer = privKeyObj.export({ type: 'pkcs8', format: 'der' })
  // PKCS8 Ed25519: DER header is 16 bytes; seed (private key) is the last 32 bytes
  const privateKey = new Uint8Array(privDer.slice(privDer.byteLength - 32))

  const pubDer = pubKeyObj.export({ type: 'spki', format: 'der' })
  // SPKI Ed25519: DER header is 12 bytes; raw key is last 32 bytes
  const publicKey = new Uint8Array(pubDer.slice(pubDer.byteLength - 32))

  const stored = {
    privateKey: Buffer.from(privateKey).toString('hex'),
    publicKey:  Buffer.from(publicKey).toString('hex'),
  }
  writeFileSync(filePath, JSON.stringify(stored, null, 2), 'utf-8')

  return { privateKey, publicKey }
}
