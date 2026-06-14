import {
  SpeculativeSession,
  type SpecConfig,
  type VerificationBackend,
} from '../../../../src/inference/speculative-session'
import type { EventSink } from '../types'

interface RunSpecPipeOptions {
  hops: number[]
  acceptanceRates: number[]
  pipelineDepths: (1 | 2)[]
  runsPerCell: number
  simHopLatencyMs: number
  sink: EventSink
}

// Fake KV client representing one peer in a chain. Per-call latency models a
// remote hop. The handler always succeeds and is depth-agnostic — the *backend*
// (PipelinedKVChain in depth=2 mode) is what makes overlap possible by scheduling
// concurrent submissions across different clients.
function makeFakeClient(hopLatencyMs: number) {
  // Wall-clock sleep via performance.now() polling. Unlike setTimeout (15 ms
  // Windows floor) and microtask chains (no real parallelism), this gives
  // sub-millisecond actual elapsed time AND lets multiple concurrent sleeps
  // observe real overlap — each polls independently and resolves once its own
  // target is reached. That is the property the sim needs: two in-flight
  // submissions at different hops complete in roughly hopLatencyMs of wall
  // clock, not 2 * hopLatencyMs.
  const sleep = (): Promise<void> => new Promise<void>(r => {
    const target = performance.now() + hopLatencyMs
    const poll = (): void => {
      if (performance.now() >= target) r()
      else setImmediate(poll)
    }
    poll()
  })
  return {
    forward: async (hidden: Float32Array, _nTokens: number, _nEmbd: number) => {
      await sleep()
      return hidden  // identity — sim doesn't care about hidden-state content
    },
    forwardAll: async (_hidden: Float32Array, _nTokens: number, _nEmbd: number) => {
      // Final-hop logits: the session will sample from these. We return a
      // synthetic sharp-at-top-token logits batch later via a separate backend
      // wrapper, because forwardAll's return shape depends on the test backend's
      // chosen vocab size. Defer to the wrapper.
      await sleep()
      return new Float32Array(0)
    },
    rollback: async (_n: number) => {},
    close: async () => {},
  }
}

// Backend wrapping a `PipelinedKVChain`-shaped chain. We don't use the real
// `PipelinedKVChain` directly because we need the *logits* returned by the chain
// to be controlled by the synthetic acceptance schedule. So we wrap a chain that
// pipelines hops the same way `PipelinedKVChain` does (per-hop tail promises),
// then patch the final-hop output to be our crafted logits.
function makeSimBackend(
  hops: number,
  hopLatencyMs: number,
  acceptanceRate: number,
  pipelineDepth: 1 | 2,
  vocabSize = 64,
): { backend: VerificationBackend; getRollbacks: () => number } {
  // Per-hop FIFO tails. Mirrors PipelinedKVChain's design.
  const hopTails: Promise<unknown>[] = Array.from({ length: hops }, () => Promise.resolve())
  const clients = Array.from({ length: hops }, () => makeFakeClient(hopLatencyMs))
  let nPast = 0
  let rollbacks = 0
  const TOP_TOKEN = 7
  // Wall-clock sleep via performance.now() polling. Unlike setTimeout (15 ms
  // Windows floor) and microtask chains (no real parallelism), this gives
  // sub-millisecond actual elapsed time AND lets multiple concurrent sleeps
  // observe real overlap — each polls independently and resolves once its own
  // target is reached. That is the property the sim needs: two in-flight
  // submissions at different hops complete in roughly hopLatencyMs of wall
  // clock, not 2 * hopLatencyMs.
  const sleep = (): Promise<void> => new Promise<void>(r => {
    const target = performance.now() + hopLatencyMs
    const poll = (): void => {
      if (performance.now() >= target) r()
      else setImmediate(poll)
    }
    poll()
  })

  function makeLogits(nTokens: number): Float32Array {
    const f = new Float32Array(nTokens * vocabSize)
    for (let t = 0; t < nTokens; t++) {
      f[t * vocabSize + TOP_TOKEN] = 10
      // For each draft slot (ids[1..]) decide accept/reject by the rate; mark
      // any non-top token with a high-but-below-margin logit when accepting.
      if (t > 0 && Math.random() < acceptanceRate) {
        for (let v = 0; v < vocabSize; v++) if (v !== TOP_TOKEN) f[t * vocabSize + v] = 9.5
      }
    }
    return f
  }

  // Single submission: passes through all hops sequentially via per-hop FIFO tails.
  // Returns once the final hop completes. With pipelineDepth=2, callers fire two
  // submissions concurrently — they overlap across hops because each hop's
  // serialization is independent.
  function doForwardAll(nTokens: number): Promise<Float32Array> {
    if (pipelineDepth === 2) {
      // Run hops sequentially through per-hop tails, returning when the chain
      // has cleared every hop. Concurrent calls overlap across hops.
      const submitChain = async (): Promise<Float32Array> => {
        let hidden = new Float32Array(0)
        for (let i = 0; i < hops - 1; i++) {
          const prevTail = hopTails[i]
          const myWork = (async () => {
            await prevTail
            return clients[i].forward(hidden, nTokens, 0)
          })()
          hopTails[i] = myWork.catch(() => {})
          hidden = await myWork
        }
        const lastIdx = hops - 1
        const prevTail = hopTails[lastIdx]
        const finalWork = (async () => {
          await prevTail
          await sleep()  // final hop's hopLatencyMs
          return makeLogits(nTokens)
        })()
        hopTails[lastIdx] = finalWork.catch(() => {})
        return finalWork
      }
      return submitChain()
    }
    // depth=1: serial — no overlap. Total wall-clock = hops * hopLatencyMs per submission.
    return (async () => {
      let hidden = new Float32Array(0)
      for (let i = 0; i < hops - 1; i++) {
        hidden = await clients[i].forward(hidden, nTokens, 0)
      }
      await sleep()  // final hop
      return makeLogits(nTokens)
    })()
  }

  const backend: VerificationBackend = {
    vocabSize,
    get nPast() { return nPast },
    set nPast(v: number) { nPast = v },
    forwardAll: (ids: Int32Array) => {
      const work = doForwardAll(ids.length)
      nPast += ids.length  // optimistic for depth=2 — mirrors PipelinedKVChain
      return work
    },
    forwardOne: (_id: number) => {
      const work = doForwardAll(1)
      nPast += 1
      return work
    },
    rollback: async (newNPast: number) => {
      rollbacks++
      await Promise.allSettled(hopTails)
      for (let i = 0; i < hops; i++) hopTails[i] = Promise.resolve()
      nPast = newNPast
    },
  }
  return { backend, getRollbacks: () => rollbacks }
}

export async function runSpecPipeSuite(opts: RunSpecPipeOptions): Promise<void> {
  const { hops, acceptanceRates, pipelineDepths, runsPerCell, simHopLatencyMs, sink } = opts
  sink({ type: 'suite:start', suite: 'spec-pipe', mode: 'sim' })

  for (const M of hops) {
    for (const accept of acceptanceRates) {
      for (const depth of pipelineDepths) {
        let totalMs = 0
        let totalTokens = 0
        let totalRollbacks = 0

        for (let run = 0; run < runsPerCell; run++) {
          const { backend, getRollbacks } = makeSimBackend(M, simHopLatencyMs, accept, depth)
          const cfg: SpecConfig = {
            enabled: true,
            ngramSize: 2,
            draftMax: 3,
            temperature: 0.01,
            topK: 1,
            marsMarginRatio: 0.9,
            adaptiveDraft: false,
            draftMin: 1,
            pipelineDepth: depth,
          }
          const session = new SpeculativeSession(backend, /*eos*/ 999, /*eot*/ undefined, cfg)
          const t0 = performance.now()
          const r = await session.generate(new Int32Array([1, 2, 3]), 32)
          totalMs += performance.now() - t0
          totalTokens += r.tokenIds.length
          totalRollbacks += getRollbacks()
        }

        const msPerToken = totalTokens > 0 ? totalMs / totalTokens : 0
        sink({
          type: 'specpipe:sample',
          pipelineDepth: depth,
          acceptanceRate: accept,
          hops: M,
          msPerToken,
          rollbacks: totalRollbacks,
          mode: 'sim',
        })
      }
    }
  }

  sink({ type: 'suite:end', suite: 'spec-pipe', mode: 'sim' })
}
