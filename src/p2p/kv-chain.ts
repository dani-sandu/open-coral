import type { KVSessionClient } from './kv-protocol'
import type { Embedder } from '../inference/embedder'
import type { VerificationBackend } from '../inference/speculative-session'

/**
 * Thrown when KVChain peers are at inconsistent positions and the session can no
 * longer be used safely. Callers must tear down the chain (close all KVSessionClients)
 * on catching this — any further operation on poisoned peers would produce wrong results.
 */
export class SessionPoisonedError extends Error {
  constructor(reason: string, public readonly cause?: unknown) {
    super(`KVChain session poisoned: ${reason}`)
    this.name = 'SessionPoisonedError'
  }
}

export class KVChain implements VerificationBackend {
  private readonly embedder: Embedder
  private readonly clients: KVSessionClient[]
  readonly vocabSize: number
  nPast = 0

  constructor(embedder: Embedder, clients: KVSessionClient[], vocabSize: number) {
    if (clients.length === 0) throw new Error('KVChain requires at least one client')
    this.embedder = embedder
    this.clients = clients
    this.vocabSize = vocabSize
  }

  async forwardAll(tokenIds: Int32Array): Promise<Float32Array> {
    const nTokens = tokenIds.length
    const { nEmbd } = this.embedder
    const preNPast = this.nPast

    // Embed is local; a failure here never touches peer state.
    let hidden = await this.embedder.embed(tokenIds)

    const advanced: KVSessionClient[] = []
    const lastIdx = this.clients.length - 1

    try {
      for (let i = 0; i < lastIdx; i++) {
        hidden = await this.clients[i].forward(hidden, nTokens, nEmbd)
        advanced.push(this.clients[i])
      }
      const logits = await this.clients[lastIdx].forwardAll(hidden, nTokens, nEmbd)
      advanced.push(this.clients[lastIdx])
      this.nPast = preNPast + nTokens
      return logits
    } catch (err) {
      await this.compensate(advanced, preNPast, err)
      throw err
    }
  }

  async forwardOne(tokenId: number): Promise<Float32Array> {
    return this.forwardAll(new Int32Array([tokenId]))
  }

  async rollback(newNPast: number): Promise<void> {
    const results = await Promise.allSettled(
      this.clients.map(c => c.rollback(newNPast)),
    )
    const failed = results.filter(r => r.status === 'rejected').length
    if (failed > 0) {
      throw new SessionPoisonedError(
        `rollback failed on ${failed}/${this.clients.length} peer(s); positions inconsistent`,
      )
    }
    this.nPast = newNPast
  }

  private async compensate(
    advanced: KVSessionClient[],
    preNPast: number,
    originalError: unknown,
  ): Promise<void> {
    if (advanced.length === 0) return
    const results = await Promise.allSettled(
      advanced.map(c => c.rollback(preNPast)),
    )
    const failed = results.filter(r => r.status === 'rejected').length
    if (failed > 0) {
      throw new SessionPoisonedError(
        `compensating rollback failed on ${failed}/${advanced.length} peer(s)`,
        originalError,
      )
    }
  }
}
