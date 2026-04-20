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

// ── embedTokens + projectToLogits ─────────────────────────────────────────────
describe('embedTokens and projectToLogits', () => {
  let handle: number

  beforeAll(() => {
    writeFileSync(MODEL_PATH, buildTinyGGUF())
    // Load full model so both embed and project paths are valid
    handle = getNative().loadBlockRange(
      MODEL_PATH, 0, TINY_CONFIG.n_blocks - 1, TINY_CONFIG.n_blocks
    )
  })

  afterAll(() => {
    getNative().freeBlockRange(handle)
    try { unlinkSync(MODEL_PATH) } catch {}
  })

  it('embedTokens returns Float32Array of length nTokens × n_embd', () => {
    const ids = new Int32Array([0, 1, 2])
    const out = getNative().embedTokens(handle, ids)
    expect(out).toBeInstanceOf(Float32Array)
    expect(out.length).toBe(ids.length * TINY_CONFIG.n_embd)
  })

  it('embedTokens output contains only finite values', () => {
    const ids = new Int32Array([0])
    const out = getNative().embedTokens(handle, ids)
    for (let i = 0; i < out.length; i++) expect(isFinite(out[i])).toBe(true)
  })

  it('projectToLogits returns Float32Array of length vocabSize', () => {
    const ids = new Int32Array([0])
    const hidden = getNative().embedTokens(handle, ids)
    const logits = getNative().projectToLogits(handle, hidden, 1)
    expect(logits).toBeInstanceOf(Float32Array)
    expect(logits.length).toBe(TINY_CONFIG.vocab_size)
  })

  it('projectToLogits output contains only finite values', () => {
    const ids = new Int32Array([0])
    const hidden = getNative().embedTokens(handle, ids)
    const logits = getNative().projectToLogits(handle, hidden, 1)
    for (let i = 0; i < logits.length; i++) expect(isFinite(logits[i])).toBe(true)
  })

  it('BlockRunner.sessionForward throws on wrong input size', () => {
    const r = new BlockRunner({
      modelPath: MODEL_PATH, blockStart: 0,
      blockEnd: TINY_CONFIG.n_blocks - 1, totalBlocks: TINY_CONFIG.n_blocks,
      hiddenSize: TINY_CONFIG.n_embd,
    })
    const sid = r.openSession(16)
    const bad = new Float32Array(TINY_CONFIG.n_embd + 1).fill(0.1)
    expect(() => r.sessionForward(sid, bad, 1)).toThrow('does not match expected')
    r.closeSession(sid)
    r.dispose()
  })

  it('BlockRunner.projectToLogits throws on wrong input size', () => {
    const r = new BlockRunner({
      modelPath: MODEL_PATH, blockStart: 0,
      blockEnd: TINY_CONFIG.n_blocks - 1, totalBlocks: TINY_CONFIG.n_blocks,
      hiddenSize: TINY_CONFIG.n_embd,
    })
    const bad = new Float32Array(TINY_CONFIG.n_embd + 1).fill(0.1)
    expect(() => r.projectToLogits(bad, 1)).toThrow('does not match expected')
    r.dispose()
  })
})

// ── Session API ───────────────────────────────────────────────────────────────
describe('session API (openSession / sessionForward / closeSession)', () => {
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

  it('openSession returns a positive integer session id', () => {
    const sid = getNative().openSession(handle, 128)
    expect(typeof sid).toBe('number')
    expect(sid).toBeGreaterThan(0)
    getNative().closeSession(handle, sid)
  })

  it('sessionForward returns Float32Array with correct length', () => {
    const sid = getNative().openSession(handle, 128)
    const input  = new Float32Array(TINY_CONFIG.n_embd).fill(0.1)
    const output = getNative().sessionForward(handle, sid, input, 1)
    expect(output).toBeInstanceOf(Float32Array)
    expect(output.length).toBe(TINY_CONFIG.n_embd)
    getNative().closeSession(handle, sid)
  })

  it('n_past advances: exceeding max_length throws on the correct call', () => {
    // Open with max_length=2 and fill it exactly; the third call must throw.
    const sid = getNative().openSession(handle, 2)
    const input = new Float32Array(TINY_CONFIG.n_embd).fill(0.1)
    getNative().sessionForward(handle, sid, input, 1) // n_past: 0 → 1
    getNative().sessionForward(handle, sid, input, 1) // n_past: 1 → 2
    expect(() => getNative().sessionForward(handle, sid, input, 1)).toThrow() // 2 + 1 > 2
    getNative().closeSession(handle, sid)
  })

  it('stateless runForward after session forward still returns finite values (seq 0 isolation)', () => {
    const sid = getNative().openSession(handle, 128)
    const hidden = new Float32Array(TINY_CONFIG.n_embd).fill(0.1)
    getNative().sessionForward(handle, sid, hidden, 1)

    // Stateless call must not be corrupted by the session's KV state
    const statelessOut = getNative().runForward(handle, hidden, 1)
    expect(statelessOut).toBeInstanceOf(Float32Array)
    for (let i = 0; i < statelessOut.length; i++) {
      expect(isFinite(statelessOut[i])).toBe(true)
    }
    getNative().closeSession(handle, sid)
  })

  it('sessionForward throws after closeSession', () => {
    const sid = getNative().openSession(handle, 128)
    getNative().closeSession(handle, sid)
    const input = new Float32Array(TINY_CONFIG.n_embd).fill(0.1)
    expect(() => getNative().sessionForward(handle, sid, input, 1)).toThrow()
  })

  it('sessions are independent: exhausting one does not affect the other', () => {
    // sid1 has max_length=1, sid2 has max_length=2.
    // Exhausting sid1 must not affect sid2's available capacity.
    const sid1 = getNative().openSession(handle, 1)
    const sid2 = getNative().openSession(handle, 2)
    const input = new Float32Array(TINY_CONFIG.n_embd).fill(0.1)

    getNative().sessionForward(handle, sid1, input, 1) // sid1 full
    expect(() => getNative().sessionForward(handle, sid1, input, 1)).toThrow() // sid1 exhausted

    // sid2 should still have its full 2-token budget untouched
    getNative().sessionForward(handle, sid2, input, 1)
    getNative().sessionForward(handle, sid2, input, 1)
    expect(() => getNative().sessionForward(handle, sid2, input, 1)).toThrow() // sid2 exhausted

    getNative().closeSession(handle, sid1)
    getNative().closeSession(handle, sid2)
  })
})
