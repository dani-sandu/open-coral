import { describe, it, expect } from 'bun:test'
import { mkdtempSync, rmSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { SimNetwork } from '../src/sim/sim-network'
import { loadOrCreateIdentity } from '../../../src/main/identity'
import { registerInferenceHandlerV3, registerInferenceHandlerV4 } from '../../../src/p2p/inference-protocol'
import { SequenceManager, type ChainStep } from '../../../src/inference/sequence-manager'

describe('pre-dial overlaps handshakes with compute', () => {
  it('a 4-hop chain pays ~one handshake on the critical path, not four', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'coral-predial-'))
    const identity = await loadOrCreateIdentity(dir)
    const nEmbd = 8
    const DIAL = 40
    const REQ = 4

    const net = new SimNetwork({ latencyMeanMs: REQ, latencyJitterMs: 0, modelBlocks: 4, dialLatencyMs: DIAL })
    const client = await net.addNode()
    const workers = []
    for (let i = 0; i < 4; i++) {
      const w = await net.addNode()
      await registerInferenceHandlerV3(w.libp2p, async (input) => input, identity)
      await registerInferenceHandlerV4(w.libp2p, async (input) => input, identity)
      workers.push(w)
    }

    const mgr = new SequenceManager({ node: client, localRunner: null, totalBlocks: 4, hiddenSize: nEmbd, identity })
    const chain: ChainStep[] = workers.map((w, i) => ({
      peerId: w.peerId, blockStart: i, blockEnd: i, multiaddr: w.multiaddrs[0],
    }))

    const input = new Float32Array(nEmbd).fill(0.5)
    const t0 = Date.now()
    const out = await mgr.runChain(chain, input, 1, 'predial')
    const elapsed = Date.now() - t0

    expect(out.length).toBe(nEmbd)
    const naiveSerial = 4 * (DIAL + REQ) // 176ms if every handshake were serial
    expect(elapsed).toBeLessThan(naiveSerial * 0.7) // < ~123ms; pre-dial should be ~56ms
    rmSync(dir, { recursive: true })
  })

  it('output is unchanged with pre-dialing (correctness)', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'coral-predial2-'))
    const identity = await loadOrCreateIdentity(dir)
    const nEmbd = 8
    const net = new SimNetwork({ latencyMeanMs: 2, latencyJitterMs: 0, modelBlocks: 2, dialLatencyMs: 10 })
    const client = await net.addNode()
    const w1 = await net.addNode()
    const w2 = await net.addNode()
    for (const w of [w1, w2]) {
      await registerInferenceHandlerV4(w.libp2p, async (input) => {
        const o = new Float32Array(input.length)
        for (let i = 0; i < input.length; i++) o[i] = input[i] * 2
        return o
      }, identity)
      await registerInferenceHandlerV3(w.libp2p, async (input) => input, identity)
    }
    const mgr = new SequenceManager({ node: client, localRunner: null, totalBlocks: 2, hiddenSize: nEmbd, identity })
    const chain: ChainStep[] = [
      { peerId: w1.peerId, blockStart: 0, blockEnd: 0, multiaddr: w1.multiaddrs[0] },
      { peerId: w2.peerId, blockStart: 1, blockEnd: 1, multiaddr: w2.multiaddrs[0] },
    ]
    const out = await mgr.runChain(chain, new Float32Array(nEmbd).fill(0.25), 1, 'pc')
    for (let i = 0; i < nEmbd; i++) expect(Math.abs(out[i] - 1.0)).toBeLessThanOrEqual(0.02) // 0.25*2*2
    rmSync(dir, { recursive: true })
  })
})
