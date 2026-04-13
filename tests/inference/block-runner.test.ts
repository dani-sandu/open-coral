import { describe, it, expect, beforeAll, afterAll } from 'bun:test'
import { writeFileSync, unlinkSync } from 'fs'
import { join } from 'path'
import { getNative } from '../../src/inference/native-loader'
import { buildTinyGGUF, TINY_CONFIG } from './fixtures/make-tiny-gguf'
import { BlockRunner } from '../../src/inference/block-runner'

const MODEL_PATH = join(import.meta.dir, 'fixtures', '_tiny-test.gguf')

// ── Addon loads ───────────────────────────────────────────────────────────────
describe('native addon', () => {
  it('loads and exposes hello()', () => {
    expect(getNative().hello()).toBe('opencoral-native ready')
  })
})

// ── Tensor loading ────────────────────────────────────────────────────────────
describe('loadBlockRange', () => {
  beforeAll(() => writeFileSync(MODEL_PATH, buildTinyGGUF()))
  afterAll(()  => { try { unlinkSync(MODEL_PATH) } catch {} })

  it('returns a positive integer handle', () => {
    const h = getNative().loadBlockRange(MODEL_PATH, 0, 0, TINY_CONFIG.n_blocks)
    expect(h).toBeGreaterThan(0)
    getNative().freeBlockRange(h)
  })

  it('loads both blocks without error', () => {
    const h = getNative().loadBlockRange(MODEL_PATH, 0, TINY_CONFIG.n_blocks - 1, TINY_CONFIG.n_blocks)
    expect(h).toBeGreaterThan(0)
    getNative().freeBlockRange(h)
  })

  it('throws on non-existent file', () => {
    expect(() =>
      getNative().loadBlockRange('/does/not/exist.gguf', 0, 0, 2)
    ).toThrow()
  })
})

// ── Forward pass ──────────────────────────────────────────────────────────────
describe('runForward', () => {
  let handle: number

  beforeAll(() => {
    writeFileSync(MODEL_PATH, buildTinyGGUF())
    handle = getNative().loadBlockRange(
      MODEL_PATH, 0, TINY_CONFIG.n_blocks - 1, TINY_CONFIG.n_blocks
    )
  })

  afterAll(() => {
    getNative().freeBlockRange(handle)
    try { unlinkSync(MODEL_PATH) } catch {}
  })

  it('returns Float32Array with correct length (n_tokens × n_embd)', () => {
    const n_tokens = 3
    const input  = new Float32Array(n_tokens * TINY_CONFIG.n_embd).fill(0.1)
    const output = getNative().runForward(handle, input, n_tokens)
    expect(output).toBeInstanceOf(Float32Array)
    expect(output.length).toBe(n_tokens * TINY_CONFIG.n_embd)
  })

  it('output contains no NaN or Infinity', () => {
    const n_tokens = 1
    const input  = new Float32Array(n_tokens * TINY_CONFIG.n_embd).fill(0.1)
    const output = getNative().runForward(handle, input, n_tokens)
    for (let i = 0; i < output.length; i++) {
      if (!isFinite(output[i])) {
        throw new Error(`output[${i}] = ${output[i]}`)
      }
    }
  })

  it('different inputs produce different outputs', () => {
    const n_tokens = 1
    const a = new Float32Array(n_tokens * TINY_CONFIG.n_embd).fill(0.1)
    const b = new Float32Array(n_tokens * TINY_CONFIG.n_embd).fill(0.9)
    const outA = getNative().runForward(handle, a, n_tokens)
    const outB = getNative().runForward(handle, b, n_tokens)
    const differs = Array.from(outA).some((v, i) => v !== outB[i])
    expect(differs).toBe(true)
  })
})

// ── BlockRunner class ─────────────────────────────────────────────────────────
describe('BlockRunner', () => {
  let runner: BlockRunner

  beforeAll(() => {
    writeFileSync(MODEL_PATH, buildTinyGGUF())
    runner = new BlockRunner({
      modelPath:   MODEL_PATH,
      blockStart:  0,
      blockEnd:    TINY_CONFIG.n_blocks - 1,
      totalBlocks: TINY_CONFIG.n_blocks,
      hiddenSize:  TINY_CONFIG.n_embd,
    })
  })

  afterAll(() => {
    runner.dispose()
    try { unlinkSync(MODEL_PATH) } catch {}
  })

  it('forward() returns Float32Array with correct size', () => {
    const n_tokens = 2
    const input  = new Float32Array(n_tokens * TINY_CONFIG.n_embd).fill(0.1)
    const output = runner.forward(input, n_tokens)
    expect(output).toBeInstanceOf(Float32Array)
    expect(output.length).toBe(n_tokens * TINY_CONFIG.n_embd)
  })

  it('forward() throws when input length is wrong', () => {
    const input = new Float32Array(TINY_CONFIG.n_embd + 1).fill(0.1)
    expect(() => runner.forward(input, 1)).toThrow('does not match expected')
  })

  it('dispose() throws on subsequent forward() call', () => {
    const r = new BlockRunner({
      modelPath:   MODEL_PATH,
      blockStart:  0,
      blockEnd:    0,
      totalBlocks: TINY_CONFIG.n_blocks,
      hiddenSize:  TINY_CONFIG.n_embd,
    })
    r.dispose()
    const input = new Float32Array(TINY_CONFIG.n_embd).fill(0.1)
    expect(() => r.forward(input, 1)).toThrow('BlockRunner has been disposed')
  })

  it('dispose() is idempotent (no error on second call)', () => {
    const r = new BlockRunner({
      modelPath:   MODEL_PATH,
      blockStart:  0,
      blockEnd:    0,
      totalBlocks: TINY_CONFIG.n_blocks,
      hiddenSize:  TINY_CONFIG.n_embd,
    })
    r.dispose()
    expect(() => r.dispose()).not.toThrow()
  })

  it('constructor throws when model file is missing', () => {
    expect(() => new BlockRunner({
      modelPath:   '/does/not/exist.gguf',
      blockStart:  0,
      blockEnd:    0,
      totalBlocks: 2,
      hiddenSize:  TINY_CONFIG.n_embd,
    })).toThrow()
  })
})
