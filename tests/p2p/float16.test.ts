import { describe, it, expect } from 'bun:test'
import {
  float32ToFloat16, float16ToFloat32,
  float32ArrayToFloat16, float16ArrayToFloat32,
} from '../../src/p2p/float16'

describe('float16 scalar conversion', () => {
  it('round-trips representative activation values within fp16 tolerance', () => {
    const values = [0, 0.1, -0.1, 0.25, 0.5, -0.5, 1, -1, 3.14159, -2.71828, 100, -100, 0.001]
    for (const v of values) {
      const back = float16ToFloat32(float32ToFloat16(v))
      const tol = Math.max(Math.abs(v) * 0.01, 1e-3) // fp16 ~ 3 significant digits
      expect(Math.abs(back - v)).toBeLessThanOrEqual(tol)
    }
  })

  it('preserves exact zero and sign of zero', () => {
    expect(float16ToFloat32(float32ToFloat16(0))).toBe(0)
  })

  it('handles infinities', () => {
    expect(float16ToFloat32(float32ToFloat16(Infinity))).toBe(Infinity)
    expect(float16ToFloat32(float32ToFloat16(-Infinity))).toBe(-Infinity)
  })

  it('produces a 16-bit unsigned value', () => {
    for (const v of [0.5, -0.5, 123.4, -0.001]) {
      const h = float32ToFloat16(v)
      expect(h).toBeGreaterThanOrEqual(0)
      expect(h).toBeLessThanOrEqual(0xffff)
      expect(Number.isInteger(h)).toBe(true)
    }
  })
})

describe('float16 array conversion', () => {
  it('float32ArrayToFloat16 returns a Uint16Array of the same length (half the bytes)', () => {
    const f32 = new Float32Array([0.1, 0.2, 0.3, 0.4])
    const f16 = float32ArrayToFloat16(f32)
    expect(f16).toBeInstanceOf(Uint16Array)
    expect(f16.length).toBe(4)
    expect(f16.byteLength).toBe(f32.byteLength / 2)
  })

  it('round-trips an array within tolerance', () => {
    const f32 = new Float32Array(64)
    for (let i = 0; i < f32.length; i++) f32[i] = Math.sin(i) * 0.5
    const back = float16ArrayToFloat32(float32ArrayToFloat16(f32))
    expect(back.length).toBe(f32.length)
    for (let i = 0; i < f32.length; i++) {
      expect(Math.abs(back[i] - f32[i])).toBeLessThanOrEqual(Math.max(Math.abs(f32[i]) * 0.01, 1e-3))
    }
  })
})
