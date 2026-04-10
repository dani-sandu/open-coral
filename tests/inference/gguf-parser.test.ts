import { describe, it, expect } from 'bun:test'
import { parseGGUFHeader } from '../../src/inference/gguf-parser'

// Build a minimal valid GGUF v3 binary in memory for testing
function buildMinimalGGUF(): Buffer {
  const parts: Buffer[] = []

  // Magic
  parts.push(Buffer.from('GGUF', 'ascii'))
  // Version = 3
  const version = Buffer.allocUnsafe(4); version.writeUInt32LE(3, 0); parts.push(version)
  // tensor_count = 2
  const tcount = Buffer.allocUnsafe(8); tcount.writeBigUInt64LE(2n, 0); parts.push(tcount)
  // metadata_kv_count = 1
  const mcount = Buffer.allocUnsafe(8); mcount.writeBigUInt64LE(1n, 0); parts.push(mcount)

  // Metadata KV: key="general.architecture", type=STRING(8), value="llama"
  const key = 'general.architecture'
  const keyLen = Buffer.allocUnsafe(8); keyLen.writeBigUInt64LE(BigInt(key.length), 0)
  parts.push(keyLen, Buffer.from(key, 'utf8'))
  const vtype = Buffer.allocUnsafe(4); vtype.writeUInt32LE(8, 0); parts.push(vtype) // STRING
  const val = 'llama'
  const valLen = Buffer.allocUnsafe(8); valLen.writeBigUInt64LE(BigInt(val.length), 0)
  parts.push(valLen, Buffer.from(val, 'utf8'))

  // Tensor 1: name="token_embd.weight", 2D [32000, 4096], type=F32(0), offset=0
  function writeTensor(name: string, dims: bigint[], ggmlType: number, offset: bigint): Buffer[] {
    const out: Buffer[] = []
    const nb = Buffer.allocUnsafe(8); nb.writeBigUInt64LE(BigInt(name.length), 0); out.push(nb)
    out.push(Buffer.from(name, 'utf8'))
    const nd = Buffer.allocUnsafe(4); nd.writeUInt32LE(dims.length, 0); out.push(nd)
    for (const d of dims) { const db = Buffer.allocUnsafe(8); db.writeBigUInt64LE(d, 0); out.push(db) }
    const tb = Buffer.allocUnsafe(4); tb.writeUInt32LE(ggmlType, 0); out.push(tb)
    const ob = Buffer.allocUnsafe(8); ob.writeBigUInt64LE(offset, 0); out.push(ob)
    return out
  }

  parts.push(...writeTensor('token_embd.weight', [32000n, 4096n], 0, 0n))
  parts.push(...writeTensor('blk.0.attn_q.weight', [4096n, 4096n], 0, 524288000n))

  // Align to 32 bytes
  const header = Buffer.concat(parts)
  const padLen = (32 - (header.length % 32)) % 32
  return Buffer.concat([header, Buffer.alloc(padLen)])
}

describe('parseGGUFHeader', () => {
  it('reads magic and version', () => {
    const buf = buildMinimalGGUF()
    const header = parseGGUFHeader(buf)
    expect(header.version).toBe(3)
  })

  it('reads tensor count', () => {
    const buf = buildMinimalGGUF()
    const header = parseGGUFHeader(buf)
    expect(header.tensorCount).toBe(2n)
  })

  it('reads metadata key-value pair', () => {
    const buf = buildMinimalGGUF()
    const header = parseGGUFHeader(buf)
    expect(header.metadata).toHaveLength(1)
    expect(header.metadata[0].key).toBe('general.architecture')
    expect(header.metadata[0].value).toBe('llama')
  })

  it('reads tensor names and shapes', () => {
    const buf = buildMinimalGGUF()
    const header = parseGGUFHeader(buf)
    expect(header.tensors).toHaveLength(2)
    expect(header.tensors[0].name).toBe('token_embd.weight')
    expect(header.tensors[0].shape).toEqual([32000n, 4096n])
    expect(header.tensors[1].name).toBe('blk.0.attn_q.weight')
  })

  it('throws on invalid magic', () => {
    const buf = buildMinimalGGUF()
    buf.write('XXXX', 0, 'ascii')
    expect(() => parseGGUFHeader(buf)).toThrow('Invalid GGUF magic')
  })

  it('throws on unsupported version', () => {
    const buf = buildMinimalGGUF()
    buf.writeUInt32LE(99, 4)
    expect(() => parseGGUFHeader(buf)).toThrow('Unsupported GGUF version')
  })

  it('computes dataRegionOffset as a multiple of 32', () => {
    const buf = buildMinimalGGUF()
    const header = parseGGUFHeader(buf)
    expect(header.dataRegionOffset % 32n).toBe(0n)
    expect(header.dataRegionOffset).toBe(BigInt(buf.length))
  })

  it('reads tensor dataOffset values', () => {
    const buf = buildMinimalGGUF()
    const header = parseGGUFHeader(buf)
    expect(header.tensors[0].dataOffset).toBe(0n)
    expect(header.tensors[1].dataOffset).toBe(524288000n)
  })

  it('reads ARRAY metadata value', () => {
    // Build a GGUF buffer with an ARRAY of 3 UINT32 values as metadata
    const parts: Buffer[] = []
    parts.push(Buffer.from('GGUF', 'ascii'))
    const version = Buffer.allocUnsafe(4); version.writeUInt32LE(3, 0); parts.push(version)
    const tcount = Buffer.allocUnsafe(8); tcount.writeBigUInt64LE(0n, 0); parts.push(tcount)
    const mcount = Buffer.allocUnsafe(8); mcount.writeBigUInt64LE(1n, 0); parts.push(mcount)

    // KV: key="arr_key", type=ARRAY(9), value=[uint32: 10, 20, 30]
    const key = 'arr_key'
    const keyLen = Buffer.allocUnsafe(8); keyLen.writeBigUInt64LE(BigInt(key.length), 0)
    parts.push(keyLen, Buffer.from(key, 'utf8'))
    const vtype = Buffer.allocUnsafe(4); vtype.writeUInt32LE(9, 0); parts.push(vtype) // ARRAY
    const elemType = Buffer.allocUnsafe(4); elemType.writeUInt32LE(4, 0); parts.push(elemType) // UINT32
    const arrLen = Buffer.allocUnsafe(8); arrLen.writeBigUInt64LE(3n, 0); parts.push(arrLen)
    for (const v of [10, 20, 30]) {
      const vb = Buffer.allocUnsafe(4); vb.writeUInt32LE(v, 0); parts.push(vb)
    }

    const header_buf = Buffer.concat(parts)
    const padLen = (32 - (header_buf.length % 32)) % 32
    const buf = Buffer.concat([header_buf, Buffer.alloc(padLen)])

    const header = parseGGUFHeader(buf)
    expect(header.metadata[0].key).toBe('arr_key')
    expect(header.metadata[0].value).toEqual([10, 20, 30])
  })
})
