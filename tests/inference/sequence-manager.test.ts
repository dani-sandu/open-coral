import { describe, it, expect, beforeAll, afterAll } from 'bun:test'
import { writeFileSync, unlinkSync, mkdtempSync, rmSync } from 'fs'
import { join } from 'path'
import { tmpdir } from 'os'
import { buildTinyGGUF, TINY_CONFIG } from './fixtures/make-tiny-gguf'
import { BlockRunner } from '../../src/inference/block-runner'
import { createOpenCoralNode, type OpenCoralNode } from '../../src/p2p/node'
import { BlockRegistry } from '../../src/p2p/block-registry'
import { registerInferenceHandlerV3 } from '../../src/p2p/inference-protocol'
import { SequenceManager, type ChainStepCandidate } from '../../src/inference/sequence-manager'
import { loadOrCreateIdentity, type NodeIdentity } from '../../src/main/identity'
import { PeerLatencyTracker } from '../../src/p2p/peer-latency'

const MODEL_PATH = join(import.meta.dir, 'fixtures', '_seq-test.gguf')

describe('SequenceManager', () => {
  let nodeA: OpenCoralNode
  let nodeB: OpenCoralNode
  let runnerA: BlockRunner   // hosts block 0
  let runnerB: BlockRunner   // hosts block 1
  let registryA: BlockRegistry
  let registryB: BlockRegistry
  let identity: NodeIdentity
  let identityDir: string

  beforeAll(async () => {
    identityDir = mkdtempSync(join(tmpdir(), 'coral-seq-test-'))
    identity = await loadOrCreateIdentity(identityDir)
    writeFileSync(MODEL_PATH, buildTinyGGUF())

    nodeA = await createOpenCoralNode()
    nodeB = await createOpenCoralNode()

    // Dial A→B so they share DHT routing tables
    await nodeA.libp2p.dial(nodeB.libp2p.getMultiaddrs()[0])
    await new Promise(r => setTimeout(r, 400))

    // Node A hosts block 0
    runnerA = new BlockRunner({
      modelPath: MODEL_PATH,
      blockStart: 0, blockEnd: 0,
      totalBlocks: TINY_CONFIG.n_blocks,
      hiddenSize: TINY_CONFIG.n_embd,
    })
    registryA = new BlockRegistry(nodeA.libp2p, 'test-org/test-model', {})
    await registryA.start()

    // Node B hosts block 1 and serves inference requests
    runnerB = new BlockRunner({
      modelPath: MODEL_PATH,
      blockStart: 1, blockEnd: 1,
      totalBlocks: TINY_CONFIG.n_blocks,
      hiddenSize: TINY_CONFIG.n_embd,
    })
    registryB = new BlockRegistry(nodeB.libp2p, 'test-org/test-model', {})
    await registryB.start()

    await registerInferenceHandlerV3(nodeB.libp2p, async (input, nTokens) =>
      runnerB.forward(input, nTokens)
    , identity)

    // Wait for DHT provider records to propagate
    await new Promise(r => setTimeout(r, 500))
  })

  afterAll(async () => {
    registryA.dispose()
    registryB.dispose()
    runnerA.dispose()
    runnerB.dispose()
    await nodeA.stop()
    await nodeB.stop()
    try { unlinkSync(MODEL_PATH) } catch {}
    rmSync(identityDir, { recursive: true })
  })

  it('planChain() finds a complete coverage across two peers', async () => {
    const peerRanges = new Map([
      [nodeB.peerId, { blockStart: 1, blockEnd: 1 }],
    ])
    const mgr = new SequenceManager({
      node: nodeA,
      localRunner: runnerA,
      totalBlocks: TINY_CONFIG.n_blocks,
      hiddenSize: TINY_CONFIG.n_embd,
      getPeerBlockRange: (peerId) => peerRanges.get(peerId) ?? null,
      identity,
    })

    const chain = await mgr.planChain()
    expect(chain.length).toBeGreaterThanOrEqual(2)
    expect(chain[0].blockStart).toBe(0)
    expect(chain[chain.length - 1].blockEnd).toBe(TINY_CONFIG.n_blocks - 1)
  })

  it('runChain() passes a tensor through both blocks and returns correct shape', async () => {
    const peerRanges = new Map([
      [nodeB.peerId, { blockStart: 1, blockEnd: 1 }],
    ])
    const mgr = new SequenceManager({
      node: nodeA,
      localRunner: runnerA,
      totalBlocks: TINY_CONFIG.n_blocks,
      hiddenSize: TINY_CONFIG.n_embd,
      getPeerBlockRange: (peerId) => peerRanges.get(peerId) ?? null,
      identity,
    })

    const chain = await mgr.planChain()
    const nTokens = 2
    const input = new Float32Array(nTokens * TINY_CONFIG.n_embd).fill(0.1)
    const output = await mgr.runChain(chain, input, nTokens)

    expect(output).toBeInstanceOf(Float32Array)
    expect(output.length).toBe(nTokens * TINY_CONFIG.n_embd)
  })

  it('runChain() output is finite and differs from input', async () => {
    const peerRanges = new Map([
      [nodeB.peerId, { blockStart: 1, blockEnd: 1 }],
    ])
    const mgr = new SequenceManager({
      node: nodeA,
      localRunner: runnerA,
      totalBlocks: TINY_CONFIG.n_blocks,
      hiddenSize: TINY_CONFIG.n_embd,
      getPeerBlockRange: (peerId) => peerRanges.get(peerId) ?? null,
      identity,
    })

    const chain = await mgr.planChain()
    const nTokens = 1
    const input = new Float32Array(nTokens * TINY_CONFIG.n_embd).fill(0.1)
    const output = await mgr.runChain(chain, input, nTokens)

    const allFinite = Array.from(output).every(v => isFinite(v))
    expect(allFinite).toBe(true)

    const differs = Array.from(output).some((v, i) => v !== input[i])
    expect(differs).toBe(true)
  })

  it('planChain() throws when no coverage is available', async () => {
    const mgr = new SequenceManager({
      node: nodeA,
      localRunner: null,
      totalBlocks: 100,
      hiddenSize: TINY_CONFIG.n_embd,
      identity,
    })

    await expect(mgr.planChain()).rejects.toThrow('No peer found')
  }, 15000)

  it('checkCoverage() reports full coverage when both blocks are hosted', async () => {
    const peerRanges = new Map([
      [nodeB.peerId, { blockStart: 1, blockEnd: 1 }],
    ])
    const mgr = new SequenceManager({
      node: nodeA,
      localRunner: runnerA,
      totalBlocks: TINY_CONFIG.n_blocks,
      hiddenSize: TINY_CONFIG.n_embd,
      getPeerBlockRange: (peerId) => peerRanges.get(peerId) ?? null,
      identity,
    })

    const report = await mgr.checkCoverage()
    expect(report.complete).toBe(true)
    expect(report.missing).toHaveLength(0)
    expect(report.covered.some(c => c.peerId === 'local')).toBe(true)
    expect(report.covered.length).toBe(2)
  })

  it('checkCoverage() reports missing when totalBlocks exceeds available coverage', async () => {
    // Block index TINY_CONFIG.n_blocks (=2) has no peer — only blocks 0 and 1 are hosted
    const peerRanges = new Map([
      [nodeB.peerId, { blockStart: 1, blockEnd: 1 }],
    ])
    const mgr = new SequenceManager({
      node: nodeA,
      localRunner: runnerA,
      totalBlocks: TINY_CONFIG.n_blocks + 1,
      hiddenSize: TINY_CONFIG.n_embd,
      getPeerBlockRange: (peerId) => peerRanges.get(peerId) ?? null,
      identity,
    })

    const report = await mgr.checkCoverage()
    expect(report.complete).toBe(false)
    expect(report.missing).toContain(TINY_CONFIG.n_blocks)
  }, 15000)
})

describe('SequenceManager — fault tolerance', () => {
  it('retries with the next candidate when the first peer throws', async () => {
    const ftIdentityDir = mkdtempSync(join(tmpdir(), 'coral-ft-test-'))
    const ftIdentity = await loadOrCreateIdentity(ftIdentityDir)

    const nodeA = await createOpenCoralNode()
    const nodeB = await createOpenCoralNode()
    await nodeA.libp2p.dial(nodeB.libp2p.getMultiaddrs()[0])

    // Both cover blocks 0–1 (TINY_CONFIG.n_blocks = 2).
    // nodeA has NO handler registered — dialing it on the inference protocol will throw.
    // nodeB has a handler that echoes input.
    await registerInferenceHandlerV3(nodeB.libp2p, async (input) => input, ftIdentity)

    const candidates: ChainStepCandidate[] = [
      { peerId: nodeA.libp2p.peerId.toString(), blockStart: 0, blockEnd: 1, multiaddr: nodeA.libp2p.getMultiaddrs()[0].toString() },
      { peerId: nodeB.libp2p.peerId.toString(), blockStart: 0, blockEnd: 1, multiaddr: nodeB.libp2p.getMultiaddrs()[0].toString() },
    ]

    const mgr = new SequenceManager({
      node: nodeA,
      localRunner: null,
      totalBlocks: 2,
      hiddenSize: TINY_CONFIG.n_embd,
      identity: ftIdentity,
    })

    const chain = await mgr.planChainWithCandidates(candidates)

    // Should succeed even though nodeA has no handler, by retrying nodeB
    const input = new Float32Array(1 * TINY_CONFIG.n_embd).fill(0.5)
    const output = await mgr.runChainWithRetry(chain, input, 1)
    expect(output).toBeInstanceOf(Float32Array)
    expect(output.length).toBe(1 * TINY_CONFIG.n_embd)

    await nodeA.stop()
    await nodeB.stop()
    rmSync(ftIdentityDir, { recursive: true })
  }, 15000)
})

describe('SequenceManager — latency-aware planning', () => {
  it('splits range when a faster peer covers part of it', async () => {
    const nodeA = await createOpenCoralNode()
    const nodeB = await createOpenCoralNode()
    const nodeC = await createOpenCoralNode()
    await nodeA.libp2p.dial(nodeB.libp2p.getMultiaddrs()[0])
    await nodeA.libp2p.dial(nodeC.libp2p.getMultiaddrs()[0])

    const identityDir2 = mkdtempSync(join(tmpdir(), 'coral-lat-test-'))
    const identity2 = await loadOrCreateIdentity(identityDir2)

    const latencyTracker = new PeerLatencyTracker()
    latencyTracker.record(nodeB.peerId, 200)
    latencyTracker.record(nodeC.peerId, 20)

    const candidates: ChainStepCandidate[] = [
      { peerId: nodeB.peerId, blockStart: 0, blockEnd: 7, multiaddr: nodeB.libp2p.getMultiaddrs()[0].toString() },
      { peerId: nodeC.peerId, blockStart: 0, blockEnd: 3, multiaddr: nodeC.libp2p.getMultiaddrs()[0].toString() },
    ]

    const mgr = new SequenceManager({
      node: nodeA,
      localRunner: null,
      totalBlocks: 8,
      hiddenSize: 16,
      latencyTracker,
      identity: identity2,
    })

    const chain = await mgr.planChainWithCandidates(candidates)

    expect(chain.length).toBe(2)
    expect(chain[0].blockStart).toBe(0)
    expect(chain[0].blockEnd).toBe(3)
    expect(chain[0].candidates[0].peerId).toBe(nodeC.peerId)
    expect(chain[1].blockStart).toBe(4)
    expect(chain[1].blockEnd).toBe(7)
    expect(chain[1].candidates[0].peerId).toBe(nodeB.peerId)

    await nodeA.stop()
    await nodeB.stop()
    await nodeC.stop()
    rmSync(identityDir2, { recursive: true })
  }, 15000)

  it('prefers continuation when latencies are similar (hop penalty)', async () => {
    const nodeA = await createOpenCoralNode()
    const nodeB = await createOpenCoralNode()
    const nodeC = await createOpenCoralNode()
    await nodeA.libp2p.dial(nodeB.libp2p.getMultiaddrs()[0])
    await nodeA.libp2p.dial(nodeC.libp2p.getMultiaddrs()[0])

    const identityDir3 = mkdtempSync(join(tmpdir(), 'coral-lat-test2-'))
    const identity3 = await loadOrCreateIdentity(identityDir3)

    const latencyTracker = new PeerLatencyTracker()
    latencyTracker.record(nodeB.peerId, 100)
    latencyTracker.record(nodeC.peerId, 90)

    const candidates: ChainStepCandidate[] = [
      { peerId: nodeB.peerId, blockStart: 0, blockEnd: 7, multiaddr: nodeB.libp2p.getMultiaddrs()[0].toString() },
      { peerId: nodeC.peerId, blockStart: 0, blockEnd: 3, multiaddr: nodeC.libp2p.getMultiaddrs()[0].toString() },
    ]

    const mgr = new SequenceManager({
      node: nodeA,
      localRunner: null,
      totalBlocks: 8,
      hiddenSize: 16,
      latencyTracker,
      identity: identity3,
    })

    const chain = await mgr.planChainWithCandidates(candidates)

    // Should NOT split — hop penalty (50ms) makes B's 100ms continuation better than C's 90ms + 50ms penalty
    expect(chain.length).toBe(1)
    expect(chain[0].blockStart).toBe(0)
    expect(chain[0].blockEnd).toBe(7)

    await nodeA.stop()
    await nodeB.stop()
    await nodeC.stop()
    rmSync(identityDir3, { recursive: true })
  }, 15000)
})
