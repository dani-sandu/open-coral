import { describe, it, expect } from 'bun:test'
import { mkdtempSync, rmSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { SimNetwork } from '../src/sim/sim-network'
import { loadOrCreateIdentity } from '../../../src/main/identity'
import { registerInferenceHandlerV3, INFERENCE_PROTOCOL_V3 } from '../../../src/p2p/inference-protocol'
import { SequenceManager, type ChainStep } from '../../../src/inference/sequence-manager'

describe('SequenceManager over SimNetwork', () => {
  it('runs a 2-hop identity chain and preserves tensor shape', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'coral-sim-seq-'))
    const identity = await loadOrCreateIdentity(dir)

    const nEmbd = 8
    const net = new SimNetwork({ latencyMeanMs: 2, latencyJitterMs: 1, modelBlocks: 2 })
    const client = await net.addNode()
    const w1 = await net.addNode()
    const w2 = await net.addNode()

    // Both workers echo input (identity forward).
    for (const w of [w1, w2]) {
      await registerInferenceHandlerV3(w.libp2p, async (input) => input, identity)
    }

    const mgr = new SequenceManager({
      node: client,
      localRunner: null,
      totalBlocks: 2,
      hiddenSize: nEmbd,
      identity,
    })

    const chain: ChainStep[] = [
      { peerId: w1.peerId, blockStart: 0, blockEnd: 0, multiaddr: w1.multiaddrs[0] },
      { peerId: w2.peerId, blockStart: 1, blockEnd: 1, multiaddr: w2.multiaddrs[0] },
    ]

    const nTokens = 4
    const input = new Float32Array(nTokens * nEmbd).fill(0.25)
    const output = await mgr.runChain(chain, input, nTokens)

    expect(output).toBeInstanceOf(Float32Array)
    expect(output.length).toBe(nTokens * nEmbd)
    for (let i = 0; i < output.length; i++) expect(output[i]).toBeCloseTo(0.25, 5)

    expect(INFERENCE_PROTOCOL_V3).toBe('/opencoral/inference/3.0.0')
    rmSync(dir, { recursive: true })
  })
})
