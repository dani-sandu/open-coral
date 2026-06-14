import { describe, it, expect } from 'bun:test'
import { mkdtempSync, rmSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { SimNetwork } from '../src/sim/sim-network'
import { loadOrCreateIdentity } from '../../../src/main/identity'
import {
  registerInferenceHandlerV3,
  sendInferenceRequestV3,
  type RemoteHopPhases,
} from '../../../src/p2p/inference-protocol'
import { peerIdFromString } from '@libp2p/peer-id'

describe('sendInferenceRequestV3 phase profiling', () => {
  it('reports four non-negative phases; injected latency dominates waitMs', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'coral-hop-phases-'))
    const identity = await loadOrCreateIdentity(dir)
    const nEmbd = 8

    // 25ms injected latency so waitMs is clearly the dominant phase.
    const net = new SimNetwork({ latencyMeanMs: 25, latencyJitterMs: 0, modelBlocks: 1 })
    const client = await net.addNode()
    const worker = await net.addNode()
    await registerInferenceHandlerV3(worker.libp2p, async (input) => input, identity)

    let captured: RemoteHopPhases | undefined
    const input = new Float32Array(1 * nEmbd).fill(0.5)
    const out = await sendInferenceRequestV3(
      client.libp2p, peerIdFromString(worker.peerId),
      input, 1, nEmbd, 'phase-req', identity,
      p => { captured = p },
    )

    expect(out).toBeInstanceOf(Float32Array)
    expect(captured).toBeDefined()
    const p = captured!
    for (const v of [p.signMs, p.sendMs, p.waitMs, p.verifyMs]) {
      expect(Number.isFinite(v)).toBe(true)
      expect(v).toBeGreaterThanOrEqual(0)
    }
    // The 25ms simulated network/remote wait should dominate the local crypto phases.
    expect(p.waitMs).toBeGreaterThan(15)
    expect(p.waitMs).toBeGreaterThan(p.signMs)
    expect(p.waitMs).toBeGreaterThan(p.verifyMs)

    rmSync(dir, { recursive: true })
  })

  it('omitting onPhases keeps the original behavior (returns the tensor)', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'coral-hop-phases2-'))
    const identity = await loadOrCreateIdentity(dir)
    const net = new SimNetwork({ latencyMeanMs: 1, latencyJitterMs: 0, modelBlocks: 1 })
    const client = await net.addNode()
    const worker = await net.addNode()
    await registerInferenceHandlerV3(worker.libp2p, async (input) => input, identity)

    const input = new Float32Array(8).fill(0.5)
    const out = await sendInferenceRequestV3(
      client.libp2p, peerIdFromString(worker.peerId), input, 1, 8, 'no-cb', identity,
    )
    expect(out.length).toBe(8)
    rmSync(dir, { recursive: true })
  })
})
