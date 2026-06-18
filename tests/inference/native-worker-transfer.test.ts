import { describe, it, expect } from 'bun:test'
import { collectInputTransfers } from '../../src/inference/native-worker'

/**
 * P2-3 zero-copy input tensor transfer.
 *
 * `collectInputTransfers(op, args)` decides which input ArrayBuffers are safe
 * to hand to `postMessage`'s transfer list (zero-copy) instead of letting the
 * structured clone copy them. The structured-clone-with-transfer calls below
 * exercise the *real* detachment mechanism postMessage uses, so they prove the
 * caller-side buffer is (or isn't) detached without spawning a worker.
 */
describe('collectInputTransfers', () => {
  it('transfers a full-span forward hidden-state tensor', () => {
    const input = new Float32Array([1, 2, 3, 4])
    const args = [0, input, 1]

    const transfer = collectInputTransfers('runForward', args)

    expect(transfer).toEqual([input.buffer])
    // Real transfer mechanism: the caller's buffer is detached afterwards.
    structuredClone({ args }, { transfer })
    expect(input.byteLength).toBe(0)
  })

  it('transfers the hidden-state tensor for sessionForward', () => {
    const input = new Float32Array(8)
    const transfer = collectInputTransfers('sessionForward', [0, 7, input, 2])

    expect(transfer).toEqual([input.buffer])
  })

  it('does NOT transfer token batches for sessionDecodeLogitsAll (caller reuses them)', () => {
    const batch = new Int32Array([10, 20, 30])

    const transfer = collectInputTransfers('sessionDecodeLogitsAll', [0, 0, batch])

    expect(transfer).toEqual([])
    structuredClone({ batch }, { transfer })
    expect(batch.length).toBe(3) // intact — SpeculativeSession re-reads batch after forwardAll
  })

  it('does NOT transfer token ids for sessionDecodeLogits', () => {
    const ids = new Int32Array([42])
    expect(collectInputTransfers('sessionDecodeLogits', [0, 0, ids])).toEqual([])
  })

  it('does NOT transfer a subarray view (would detach the shared parent buffer)', () => {
    const parent = new Float32Array(8)
    const view = parent.subarray(2, 6)

    const transfer = collectInputTransfers('runForward', [0, view, 4])

    expect(transfer).toEqual([])
    structuredClone({ view }, { transfer })
    expect(parent.byteLength).toBe(32) // parent (re-prefill buffer) untouched
  })

  it('ignores non-view arguments (handles, token counts)', () => {
    const input = new Float32Array(4)
    const transfer = collectInputTransfers('runForward', [5, input, 1])
    expect(transfer).toEqual([input.buffer])
  })

  it('deduplicates when the same buffer appears twice', () => {
    const input = new Float32Array(4)
    const view = new Float32Array(input.buffer)
    const transfer = collectInputTransfers('runForward', [0, input, view])
    expect(transfer).toEqual([input.buffer])
  })
})
