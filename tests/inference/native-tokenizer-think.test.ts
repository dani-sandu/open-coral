import { describe, it, expect } from 'bun:test'
import { pickSingleToken } from '../../src/inference/native-tokenizer'

describe('pickSingleToken', () => {
  it('returns the id when the input is exactly one token', () => {
    expect(pickSingleToken(new Int32Array([42]))).toBe(42)
  })
  it('returns undefined when the input is multiple tokens', () => {
    expect(pickSingleToken(new Int32Array([1, 2, 3]))).toBeUndefined()
  })
  it('returns undefined when the input is empty', () => {
    expect(pickSingleToken(new Int32Array(0))).toBeUndefined()
  })
})
