/**
 * Polyfill for crypto.hash() — a Node 21.7+ API not yet supported by Bun.
 * Vite 7 uses crypto.hash() internally for dependency hashing.
 * This shim delegates to the widely-supported crypto.createHash().
 */
import crypto from 'node:crypto'

if (typeof (crypto as any).hash !== 'function') {
  ;(crypto as any).hash = (
    algorithm: string,
    data: crypto.BinaryLike,
    outputEncoding?: crypto.BinaryToTextEncoding,
  ): string | Buffer => {
    const h = crypto.createHash(algorithm).update(data)
    return outputEncoding ? h.digest(outputEncoding) : h.digest()
  }
}
