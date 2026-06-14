import { describe, it, expect } from 'bun:test'
import { mkdtempSync, rmSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { SimNetwork } from '../src/sim/sim-network'
import { loadOrCreateIdentity } from '../../../src/main/identity'
import { registerInferenceHandlerV3 } from '../../../src/p2p/inference-protocol'
import { SequenceManager, type ChainStep, type HopTiming } from '../../../src/inference/sequence-manager'

describe('runChain hopLog', () => {
  it('records one HopTiming per step: local has forwardMs, remote has phases', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'coral-hop-log-'))
    const identity = await loadOrCreateIdentity(dir)
    const nEmbd = 8

    const net = new SimNetwork({ latencyMeanMs: 10, latencyJitterMs: 0, modelBlocks: 2 })
    const client = await net.addNode()
    const worker = await net.addNode()
    await registerInferenceHandlerV3(worker.libp2p, async (input) => input, identity)

    // A local step (identity forward) followed by a remote step.
    const localRunner = {
      blockStart: 0, blockEnd: 0,
      forward: (x: Float32Array) => x,
    }
    const mgr = new SequenceManager({
      node: client, localRunner, totalBlocks: 2, hiddenSize: nEmbd, identity,
    })

    const chain: ChainStep[] = [
      { peerId: 'local', blockStart: 0, blockEnd: 0 },
      { peerId: worker.peerId, blockStart: 1, blockEnd: 1, multiaddr: worker.multiaddrs[0] },
    ]

    const hopLog: HopTiming[] = []
    const input = new Float32Array(1 * nEmbd).fill(0.25)
    const out = await mgr.runChain(chain, input, 1, 'hl-req', hopLog)

    expect(out.length).toBe(nEmbd)
    expect(hopLog.length).toBe(2)

    // Local hop
    expect(hopLog[0].peerId).toBe('local')
    expect(hopLog[0].forwardMs).toBeGreaterThanOrEqual(0)
    expect(hopLog[0].phases).toBeUndefined()

    // Remote hop
    expect(hopLog[1].peerId).toBe(worker.peerId)
    expect(hopLog[1].forwardMs).toBeUndefined()
    expect(hopLog[1].phases).toBeDefined()
    const p = hopLog[1].phases!
    const sum = p.signMs + p.sendMs + p.waitMs + p.verifyMs
    // Phases account for ~all of the remote hop's total (dial is ~0 in-process).
    expect(sum).toBeLessThanOrEqual(hopLog[1].totalMs + 1)
    expect(hopLog[1].totalMs - sum).toBeLessThan(5)

    rmSync(dir, { recursive: true })
  })

  it('runChain without a hopLog behaves exactly as before', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'coral-hop-log2-'))
    const identity = await loadOrCreateIdentity(dir)
    const net = new SimNetwork({ latencyMeanMs: 1, latencyJitterMs: 0, modelBlocks: 1 })
    const client = await net.addNode()
    const worker = await net.addNode()
    await registerInferenceHandlerV3(worker.libp2p, async (input) => input, identity)
    const mgr = new SequenceManager({ node: client, localRunner: null, totalBlocks: 1, hiddenSize: 8, identity })
    const chain: ChainStep[] = [{ peerId: worker.peerId, blockStart: 0, blockEnd: 0, multiaddr: worker.multiaddrs[0] }]
    const out = await mgr.runChain(chain, new Float32Array(8).fill(0.1), 1)
    expect(out.length).toBe(8)
    rmSync(dir, { recursive: true })
  })
})
