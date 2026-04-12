import { describe, it, expect } from 'bun:test'
import { extractBlockTensors } from '../../src/inference/block-extractor'
import { GGUFHeader, GGUFTensorInfo, GGMLType, GGUFValueType } from '../../src/inference/types'

function makeTensor(name: string): GGUFTensorInfo {
  return { name, shape: [4096n, 4096n], type: GGMLType.Q4_K, dataOffset: 0n }
}

function makeHeader(tensorNames: string[]): GGUFHeader {
  return {
    version: 3,
    tensorCount: BigInt(tensorNames.length),
    metadata: [{ key: 'general.architecture', valueType: GGUFValueType.STRING, value: 'llama' }],
    tensors: tensorNames.map(makeTensor),
    metadataEndOffset: 100,
    dataRegionOffset: 1024n
  }
}

describe('extractBlockTensors', () => {
  const tensorNames = [
    'token_embd.weight',
    'blk.0.attn_q.weight', 'blk.0.attn_k.weight', 'blk.0.ffn_gate.weight',
    'blk.1.attn_q.weight', 'blk.1.attn_k.weight', 'blk.1.ffn_gate.weight',
    'blk.2.attn_q.weight', 'blk.2.attn_k.weight', 'blk.2.ffn_gate.weight',
    'output_norm.weight', 'output.weight'
  ]

  it('extracts tensors for middle block range (no embedding, no output)', () => {
    const header = makeHeader(tensorNames)
    const result = extractBlockTensors(header, { start: 1, end: 2 })
    expect(result.blockTensors.map(t => t.name)).toEqual([
      'blk.1.attn_q.weight', 'blk.1.attn_k.weight', 'blk.1.ffn_gate.weight',
      'blk.2.attn_q.weight', 'blk.2.attn_k.weight', 'blk.2.ffn_gate.weight'
    ])
    expect(result.embeddingTensor).toBeNull()
    expect(result.outputTensors).toHaveLength(0)
  })

  it('includes embedding tensor when range starts at block 0', () => {
    const header = makeHeader(tensorNames)
    const result = extractBlockTensors(header, { start: 0, end: 1 })
    expect(result.embeddingTensor?.name).toBe('token_embd.weight')
    expect(result.blockTensors.map(t => t.name)).toContain('blk.0.attn_q.weight')
    expect(result.blockTensors.map(t => t.name)).toContain('blk.1.attn_q.weight')
  })

  it('includes output tensors when total block count matches model', () => {
    const header = makeHeader(tensorNames)
    // 3 blocks total (0,1,2), range covers last block but not block 0.
    // token_embd.weight is also included as a weight-tying fallback for
    // models where output.weight doesn't exist (e.g. Gemma).
    const result = extractBlockTensors(header, { start: 2, end: 2 }, 3)
    expect(result.outputTensors.map(t => t.name)).toEqual([
      'token_embd.weight', 'output_norm.weight', 'output.weight',
    ])
  })

  it('does not include token_embd in outputTensors when also hosting block 0', () => {
    const header = makeHeader(tensorNames)
    // Range covers both first and last block — embedding goes to embeddingTensor, not outputTensors
    const result = extractBlockTensors(header, { start: 0, end: 2 }, 3)
    expect(result.embeddingTensor?.name).toBe('token_embd.weight')
    expect(result.outputTensors.map(t => t.name)).toEqual(['output_norm.weight', 'output.weight'])
  })

  it('returns correct range on result', () => {
    const header = makeHeader(tensorNames)
    const result = extractBlockTensors(header, { start: 1, end: 2 })
    expect(result.range).toEqual({ start: 1, end: 2 })
  })

  it('throws when range end exceeds available blocks', () => {
    const header = makeHeader(tensorNames)
    expect(() => extractBlockTensors(header, { start: 0, end: 99 })).toThrow('Block range [0..99] exceeds')
  })
})
