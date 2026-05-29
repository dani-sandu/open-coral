// IEEE 754 half-precision conversion (round-to-nearest). Scratch views are reused.
const f32 = new Float32Array(1)
const u32 = new Uint32Array(f32.buffer)

/** Convert a JS number to a Float16 bit pattern (uint16, 0..0xffff). */
export function float32ToFloat16(val: number): number {
  f32[0] = val
  const x = u32[0]
  const sign = (x >>> 16) & 0x8000
  const e = (x >>> 23) & 0xff
  let m = x & 0x7fffff

  if (e === 0xff) {
    // Inf or NaN
    return sign | 0x7c00 | (m ? 0x0200 : 0)
  }

  // Rebias exponent from 127 (single) to 15 (half).
  const he = e - 112
  if (he >= 0x1f) {
    return sign | 0x7c00 // overflow → inf
  }
  if (he <= 0) {
    if (he < -10) return sign // underflow → signed zero
    m |= 0x800000 // restore implicit leading 1
    const shift = 14 - he // 14..24
    let half = m >>> shift
    if ((m >>> (shift - 1)) & 1) half += 1 // round to nearest
    return sign | half
  }
  // Normal half. A rounding carry naturally propagates into the exponent bits.
  let half = (he << 10) | (m >>> 13)
  if ((m >>> 12) & 1) half += 1
  return sign | half
}

/** Convert a Float16 bit pattern (uint16) back to a JS number. */
export function float16ToFloat32(h: number): number {
  const sign = (h & 0x8000) ? -1 : 1
  const e = (h & 0x7c00) >>> 10
  const m = h & 0x03ff
  if (e === 0) return sign * Math.pow(2, -14) * (m / 1024)
  if (e === 0x1f) return m ? NaN : sign * Infinity
  return sign * Math.pow(2, e - 15) * (1 + m / 1024)
}

export function float32ArrayToFloat16(arr: Float32Array): Uint16Array {
  const out = new Uint16Array(arr.length)
  for (let i = 0; i < arr.length; i++) out[i] = float32ToFloat16(arr[i])
  return out
}

export function float16ArrayToFloat32(arr: Uint16Array): Float32Array {
  const out = new Float32Array(arr.length)
  for (let i = 0; i < arr.length; i++) out[i] = float16ToFloat32(arr[i])
  return out
}
