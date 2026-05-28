import { describe, it, expect } from 'bun:test'
import { mkdtempSync, rmSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { SimNetwork } from '../src/sim/sim-network'
import { loadOrCreateIdentity, type NodeIdentity } from '../../../src/main/identity'
import { registerInferenceHandlerV3, sendInferenceRequestV3 } from '../../../src/p2p/inference-protocol'
import { peerIdFromString } from '@libp2p/peer-id'

describe('SimNetwork latency sampling', () => {
  it('samples are non-negative and clamped to [0, mean + 5*jitter]', () => {
    const net = new SimNetwork({ latencyMeanMs: 20, latencyJitterMs: 8, modelBlocks: 4 })
    for (let i = 0; i < 1000; i++) {
      const s = net.sampleLatency()
      expect(s).toBeGreaterThanOrEqual(0)
      expect(s).toBeLessThanOrEqual(20 + 5 * 8)
    }
  })
})

describe('SimNetwork inference round-trip', () => {
  let identityDir: string
  let identity: NodeIdentity

  it('routes a real V3 inference request through the in-process handler', async () => {
    identityDir = mkdtempSync(join(tmpdir(), 'coral-sim-net-'))
    identity = await loadOrCreateIdentity(identityDir)

    // Near-zero latency keeps the test fast.
    const net = new SimNetwork({ latencyMeanMs: 1, latencyJitterMs: 0, modelBlocks: 4 })
    const client = await net.addNode()
    const worker = await net.addNode()

    const nEmbd = 8
    // Worker hosts an inference handler that scales the tensor by 2.
    await registerInferenceHandlerV3(worker.libp2p, async (input) => {
      const out = new Float32Array(input.length)
      for (let i = 0; i < input.length; i++) out[i] = input[i] * 2
      return out
    }, identity)

    const input = new Float32Array(1 * nEmbd).fill(0.5)
    const output = await sendInferenceRequestV3(
      client.libp2p,
      peerIdFromString(worker.peerId),
      input, 1, nEmbd, 'sim-req-1', identity,
    )

    expect(output).toBeInstanceOf(Float32Array)
    expect(output.length).toBe(nEmbd)
    for (let i = 0; i < output.length; i++) expect(output[i]).toBeCloseTo(1.0, 5)

    rmSync(identityDir, { recursive: true })
  })
})
