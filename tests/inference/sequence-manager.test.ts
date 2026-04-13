import { describe, it, expect, beforeAll, afterAll } from 'bun:test'
import { writeFileSync, unlinkSync } from 'fs'
import { join } from 'path'
import { buildTinyGGUF, TINY_CONFIG } from './fixtures/make-tiny-gguf'
import { BlockRunner } from '../../src/inference/block-runner'
import { createOpenCoralNode, type OpenCoralNode } from '../../src/p2p/node'
import { BlockRegistry } from '../../src/p2p/block-registry'
import { registerInferenceHandler } from '../../src/p2p/inference-protocol'
import { SequenceManager, type ChainStepCandidate } from '../../src/inference/sequence-manager'

const MODEL_PATH = join(import.meta.dir, 'fixtures', '_seq-test.gguf')

describe('SequenceManager', () => {
  let nodeA: OpenCoralNode
  let nodeB: OpenCoralNode
  let runnerA: BlockRunner   // hosts block 0
  let runnerB: BlockRunner   // hosts block 1
  let registryA: BlockRegistry
  let registryB: BlockRegistry

  beforeAll(async () => {
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
    registryA = new BlockRegistry(nodeA.libp2p, {})
    await registryA.start()

    // Node B hosts block 1 and serves inference requests
    runnerB = new BlockRunner({
      modelPath: MODEL_PATH,
      blockStart: 1, blockEnd: 1,
      totalBlocks: TINY_CONFIG.n_blocks,
      hiddenSize: TINY_CONFIG.n_embd,
    })
    registryB = new BlockRegistry(nodeB.libp2p, {})
    await registryB.start()

    await registerInferenceHandler(nodeB.libp2p, async (input, nTokens) =>
      runnerB.forward(input, nTokens)
    )

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
    })

    const report = await mgr.checkCoverage()
    expect(report.complete).toBe(false)
    expect(report.missing).toContain(TINY_CONFIG.n_blocks)
  }, 15000)
})

describe('SequenceManager — fault tolerance', () => {
  it('retries with the next candidate when the first peer throws', async () => {
    const nodeA = await createOpenCoralNode()
    const nodeB = await createOpenCoralNode()
    await nodeA.libp2p.dial(nodeB.libp2p.getMultiaddrs()[0])

    // Both cover blocks 0–1 (TINY_CONFIG.n_blocks = 2).
    // nodeA has NO handler registered — dialing it on the inference protocol will throw.
    // nodeB has a handler that echoes input.
    await registerInferenceHandler(nodeB.libp2p, async (input) => input)

    const candidates: ChainStepCandidate[] = [
      { peerId: nodeA.libp2p.peerId.toString(), blockStart: 0, blockEnd: 1, multiaddr: nodeA.libp2p.getMultiaddrs()[0].toString() },
      { peerId: nodeB.libp2p.peerId.toString(), blockStart: 0, blockEnd: 1, multiaddr: nodeB.libp2p.getMultiaddrs()[0].toString() },
    ]

    const mgr = new SequenceManager({
      node: nodeA,
      localRunner: null,
      totalBlocks: 2,
      hiddenSize: TINY_CONFIG.n_embd,
    })

    const chain = await mgr.planChainWithCandidates(candidates)

    // Should succeed even though nodeA has no handler, by retrying nodeB
    const input = new Float32Array(1 * TINY_CONFIG.n_embd).fill(0.5)
    const output = await mgr.runChainWithRetry(chain, input, 1)
    expect(output).toBeInstanceOf(Float32Array)
    expect(output.length).toBe(1 * TINY_CONFIG.n_embd)

    await nodeA.stop()
    await nodeB.stop()
  }, 15000)
})
