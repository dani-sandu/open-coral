import type { KVChain } from './kv-chain'
import type { VerificationBackend } from '../inference/speculative-session'

/**
 * Wraps a KVChain with optional P=2 optimistic pipelining (SpecPipe).
 *
 * - depth=1: pure passthrough to KVChain. Byte-identical to today.
 * - depth=2: TODO in Task 3 — per-hop tail-promise pipelining so step N+1 can
 *   submit to hop 0 while step N is at hop k>0.
 *
 * See docs/superpowers/specs/2026-06-05-p2-12-specpipe-pipelined-speculation-design.md
 */
export class PipelinedKVChain implements VerificationBackend {
  readonly vocabSize: number
  nPast = 0

  // Per-hop tail promise — each new submission's call at hop i awaits the previous
  // submission's call at hop i. This serializes per-client (mandatory because
  // KVSessionClient is not concurrent-safe — see kv-protocol.ts) while letting
  // step N+1 occupy hop 0 while step N is at hop 1, etc.
  private readonly hopTails: Promise<unknown>[] = []

  // Top-level in-flight forwardAll promises registered synchronously (before any
  // await) so rollback can gate even when hopTails hasn't been fully populated yet.
  // Snapshot is taken by Promise.allSettled at rollback time; cleanup of this set
  // is registered as a .then on the individual promises.
  private readonly inFlight = new Set<Promise<unknown>>()

  constructor(
    private readonly chain: KVChain,
    readonly pipelineDepth: 1 | 2,
  ) {
    this.vocabSize = chain.vocabSize
    this.nPast = chain.nPast
  }

  // NOT async: returning the work promise directly ensures callers' `.then` fires
  // in the same microtask cycle as the internal rollback-gating handler. Wrapping
  // in `async` would chain through an extra microtask, which lets a follow-up
  // rollback's body run before the caller's observer of the resolved forward —
  // breaking the per-submission ordering guarantee that the gating relies on.
  forwardAll(batch: Int32Array): Promise<Float32Array> {
    if (this.pipelineDepth === 1) {
      return this.chain.forwardAll(batch).then(logits => {
        this.nPast = this.chain.nPast
        return logits
      })
    }

    // depth=2: schedule across hops with per-hop FIFO tails.
    const { clients, embedder } = this.chain
    if (clients.length === 0) throw new Error('PipelinedKVChain: empty clients')
    const nEmbd = embedder.nEmbd
    const nTokens = batch.length

    // Optimistic nPast: advance now, callers' rollback math reads this getter.
    this.nPast += nTokens

    if (this.hopTails.length !== clients.length) {
      // Lazy init on first submission so we know clients.length.
      this.hopTails.length = 0
      for (let i = 0; i < clients.length; i++) this.hopTails.push(Promise.resolve())
    }

    // Dispatch the pipelined work. The returned promise (`work`) resolves when
    // the full pipeline completes. We register it into `inFlight` synchronously
    // here (before any await in forwardAll itself) so rollback can gate on it
    // regardless of where in the pipeline we are.
    const work = this._doForwardAll(batch, clients, embedder, nEmbd, nTokens)
    this.inFlight.add(work)
    work.then(() => this.inFlight.delete(work), () => this.inFlight.delete(work))
    return work
  }

  private async _doForwardAll(
    batch: Int32Array,
    clients: KVChain['clients'],
    embedder: KVChain['embedder'],
    nEmbd: number,
    nTokens: number,
  ): Promise<Float32Array> {
    // Stage 0: local embedding (no hop tail — purely client-side).
    let hidden = await embedder.embed(batch)

    const lastIdx = clients.length - 1
    for (let i = 0; i < lastIdx; i++) {
      const prevTail = this.hopTails[i]
      // Snapshot `hidden` — it is reassigned by `await myWork` later in this loop
      // iteration, so the closure must capture the current value rather than read
      // the mutated outer variable.
      const localHidden = hidden
      const myWork = (async () => {
        await prevTail
        return clients[i].forward(localHidden, nTokens, nEmbd)
      })()
      // Next submission at this hop waits for *our* forward to complete.
      this.hopTails[i] = myWork.catch(() => {})  // suppress unhandled-rejection on tail chain
      hidden = await myWork
    }

    const prevTail = this.hopTails[lastIdx]
    const finalWork = (async () => {
      await prevTail
      return clients[lastIdx].forwardAll(hidden, nTokens, nEmbd)
    })()
    this.hopTails[lastIdx] = finalWork.catch(() => {})
    return finalWork
  }

  async forwardOne(tokenId: number): Promise<Float32Array> {
    return this.forwardAll(new Int32Array([tokenId]))
  }

  async rollback(newNPast: number): Promise<void> {
    // Gate the whole pipeline: every in-flight forward must drain before rollback
    // hits the peers. Otherwise rollback could arrive before a still-in-flight
    // write lands, leaving the peer's KV ahead of our intended position.
    // We gate on `inFlight` (registered synchronously at forwardAll call-time)
    // rather than `hopTails` alone, because hopTails entries are replaced with
    // real work only after the embed await inside _doForwardAll, so hopTails may
    // still hold Promise.resolve() sentinels when rollback is called.
    if (this.inFlight.size > 0) {
      await Promise.allSettled(this.inFlight)
    }
    if (this.hopTails.length > 0) {
      for (let i = 0; i < this.hopTails.length; i++) this.hopTails[i] = Promise.resolve()
    }
    await this.chain.rollback(newNPast)
    this.nPast = newNPast
  }
}
