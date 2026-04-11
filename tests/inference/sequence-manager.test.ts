import { describe, it, expect, beforeAll, afterAll } from 'bun:test'
import { writeFileSync, unlinkSync } from 'fs'
import { join } from 'path'
import { buildTinyGGUF, TINY_CONFIG } from './fixtures/make-tiny-gguf'
import { BlockRunner } from '../../src/inference/block-runner'
import { createCoralNode, type CoralNode } from '../../src/p2p/node'
import { BlockRegistry } from '../../src/p2p/block-registry'
import { registerInferenceHandler } from '../../src/p2p/inference-protocol'
import { SequenceManager } from '../../src/inference/sequence-manager'

const MODEL_PATH = join(import.meta.dir, 'fixtures', '_seq-test.gguf')

describe('SequenceManager', () => {
  let nodeA: CoralNode
  let nodeB: CoralNode
  let runnerA: BlockRunner   // hosts block 0
  let runnerB: BlockRunner   // hosts block 1
  let registryA: BlockRegistry
  let registryB: BlockRegistry

  beforeAll(async () => {
    writeFileSync(MODEL_PATH, buildTinyGGUF())

    nodeA = await createCoralNode()
    nodeB = await createCoralNode()

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
    registryA = new BlockRegistry(nodeA.libp2p, { blockStart: 0, blockEnd: 0 })
    await registryA.start()

    // Node B hosts block 1 and serves inference requests
    runnerB = new BlockRunner({
      modelPath: MODEL_PATH,
      blockStart: 1, blockEnd: 1,
      totalBlocks: TINY_CONFIG.n_blocks,
      hiddenSize: TINY_CONFIG.n_embd,
    })
    registryB = new BlockRegistry(nodeB.libp2p, { blockStart: 1, blockEnd: 1 })
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
    const mgr = new SequenceManager({
      node: nodeA,
      localRunner: runnerA,
      totalBlocks: TINY_CONFIG.n_blocks,
      hiddenSize: TINY_CONFIG.n_embd,
    })

    const chain = await mgr.planChain()
    expect(chain.length).toBeGreaterThanOrEqual(2)
    expect(chain[0].blockStart).toBe(0)
    expect(chain[chain.length - 1].blockEnd).toBe(TINY_CONFIG.n_blocks - 1)
  })

  it('runChain() passes a tensor through both blocks and returns correct shape', async () => {
    const mgr = new SequenceManager({
      node: nodeA,
      localRunner: runnerA,
      totalBlocks: TINY_CONFIG.n_blocks,
      hiddenSize: TINY_CONFIG.n_embd,
    })

    const chain = await mgr.planChain()
    const nTokens = 2
    const input = new Float32Array(nTokens * TINY_CONFIG.n_embd).fill(0.1)
    const output = await mgr.runChain(chain, input, nTokens)

    expect(output).toBeInstanceOf(Float32Array)
    expect(output.length).toBe(nTokens * TINY_CONFIG.n_embd)
  })

  it('runChain() output is finite and differs from input', async () => {
    const mgr = new SequenceManager({
      node: nodeA,
      localRunner: runnerA,
      totalBlocks: TINY_CONFIG.n_blocks,
      hiddenSize: TINY_CONFIG.n_embd,
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
  })
})
