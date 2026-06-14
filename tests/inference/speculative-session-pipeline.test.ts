import { describe, it, expect } from 'bun:test'
import {
  SpeculativeSession,
  LocalVerificationBackend,
  type SpecConfig,
  type VerificationBackend,
} from '../../src/inference/speculative-session'
import type { AsyncBlockRunner } from '../../src/inference/native-worker'

// ── Deterministic synthetic backend ──────────────────────────────────────────
//
// Produces logits where draftTokens[i] is at a controlled "relative score" to
// the top token: when scoreRatio >= marsMarginRatio, MARS deterministically
// accepts the draft. We choose scores per-token via a precomputed accept
// schedule so test cases pin specific accept/reject patterns.

const VOCAB = 32
const TOP_TOKEN = 7  // arbitrary "top" choice; not the draft so MARS path triggers

function sharpAt(token: number, n: number): Float32Array {
  // Logits batch of `n` positions; each position's top is TOP_TOKEN at +10.
  const f = new Float32Array(n * VOCAB)
  for (let t = 0; t < n; t++) f[t * VOCAB + token] = 10
  return f
}

function makeAcceptingBackend(acceptSchedule: boolean[]): {
  backend: VerificationBackend
  callLog: { kind: 'all' | 'one'; nPast: number; len: number }[]
} {
  const callLog: { kind: 'all' | 'one'; nPast: number; len: number }[] = []
  let nPast = 0
  const backend: VerificationBackend = {
    vocabSize: VOCAB,
    get nPast() { return nPast },
    set nPast(v: number) { nPast = v },
    forwardAll: async (ids: Int32Array) => {
      callLog.push({ kind: 'all', nPast, len: ids.length })
      // Accept decisions are keyed by KV POSITION rather than a call counter so
      // a pre-submitted-then-rolled-back forwardAll (P2-12 SpecPipe depth=2)
      // doesn't shift the schedule that the canonical re-submission sees.
      const out = sharpAt(TOP_TOKEN, ids.length)
      for (let i = 1; i < ids.length; i++) {
        const accept = acceptSchedule[(nPast + i) % acceptSchedule.length]
        const slotLogit = accept ? 10 : -10
        out[(i - 1) * VOCAB + ids[i]] = slotLogit
      }
      nPast += ids.length
      return out
    },
    forwardOne: async (id: number) => {
      callLog.push({ kind: 'one', nPast, len: 1 })
      nPast += 1
      return sharpAt(TOP_TOKEN, 1)
    },
    rollback: async (newNPast: number) => { nPast = newNPast },
  }
  return { backend, callLog }
}

// ── The property test ───────────────────────────────────────────────────────

async function runWithDepth(depth: 1 | 2, acceptSchedule: boolean[]): Promise<number[]> {
  const { backend } = makeAcceptingBackend(acceptSchedule)
  const cfg: SpecConfig = {
    enabled: true,
    ngramSize: 2,
    draftMax: 3,
    temperature: 0.01,   // near-deterministic top-k sample
    topK: 1,
    marsMarginRatio: 0.9,
    adaptiveDraft: false,
    draftMin: 1,
    pipelineDepth: depth,
  }
  const session = new SpeculativeSession(backend, /*eos*/ 999, /*eot*/ undefined, cfg)
  const result = await session.generate(new Int32Array([1, 2, 3]), 16)
  return result.tokenIds
}

describe('SpeculativeSession — pipelineDepth output preservation', () => {
  const schedules: { name: string; schedule: boolean[] }[] = [
    { name: 'all-accept (0.9)', schedule: Array(64).fill(true) },
    { name: 'mixed (0.6)',     schedule: Array.from({ length: 64 }, (_, i) => i % 5 !== 0) },
    { name: 'mostly-reject (0.3)', schedule: Array.from({ length: 64 }, (_, i) => i % 3 === 0) },
  ]

  for (const { name, schedule } of schedules) {
    it(`identical token output across depth=1 and depth=2 — ${name}`, async () => {
      const a = await runWithDepth(1, schedule)
      const b = await runWithDepth(2, schedule)
      expect(b).toEqual(a)
    })
  }
})
