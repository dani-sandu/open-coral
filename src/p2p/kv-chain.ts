import type { KVSessionClient } from './kv-protocol'
import type { VerificationBackend } from '../inference/speculative-session'

type EmbedFn = (tokenIds: Int32Array) => Promise<Float32Array>

export class KVChain implements VerificationBackend {
  private readonly embed: EmbedFn
  private readonly clients: KVSessionClient[]
  readonly vocabSize: number
  nPast = 0

  constructor(embed: EmbedFn, clients: KVSessionClient[], vocabSize: number) {
    if (clients.length === 0) throw new Error('KVChain requires at least one client')
    this.embed = embed
    this.clients = clients
    this.vocabSize = vocabSize
  }

  async forwardAll(tokenIds: Int32Array): Promise<Float32Array> {
    const nTokens = tokenIds.length
    let hidden = await this.embed(tokenIds)
    const nEmbd = hidden.length / nTokens

    // Forward through all peers except the last (they return hidden states)
    for (let i = 0; i < this.clients.length - 1; i++) {
      hidden = await this.clients[i].forward(hidden, nTokens, nEmbd)
    }

    // Last peer: returns logits for all token positions
    const logits = await this.clients[this.clients.length - 1].forwardAll(hidden, nTokens, nEmbd)
    this.nPast += nTokens
    return logits
  }

  async forwardOne(tokenId: number): Promise<Float32Array> {
    return this.forwardAll(new Int32Array([tokenId]))
  }

  async rollback(newNPast: number): Promise<void> {
    await Promise.all(this.clients.map(c => c.rollback(newNPast)))
    this.nPast = newNPast
  }
}
