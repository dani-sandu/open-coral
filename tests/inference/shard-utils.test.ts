import { describe, it, expect } from 'bun:test'
import { parseShardInfo, isShardSeed } from '../../src/inference/shard-utils'

describe('parseShardInfo', () => {
  it('parses a shard filename', () => {
    expect(parseShardInfo('GLM-5.1-UD-Q4_K_M-00001-of-00011.gguf')).toEqual({
      prefix: 'GLM-5.1-UD-Q4_K_M',
      shardNo: 1,
      totalShards: 11,
    })
  })

  it('parses shard 7 of 11', () => {
    expect(parseShardInfo('GLM-5.1-UD-Q4_K_M-00007-of-00011.gguf')).toEqual({
      prefix: 'GLM-5.1-UD-Q4_K_M',
      shardNo: 7,
      totalShards: 11,
    })
  })

  it('returns null for a non-shard filename', () => {
    expect(parseShardInfo('llama-3-8b-Q4_K_M.gguf')).toBeNull()
  })

  it('returns null for a partial GGUF (blocks suffix)', () => {
    expect(parseShardInfo('model.blocks-0-3.gguf')).toBeNull()
  })
})

describe('isShardSeed', () => {
  it('returns true for shard 1', () => {
    expect(isShardSeed('GLM-5.1-UD-Q4_K_M-00001-of-00011.gguf')).toBe(true)
  })

  it('returns false for shard 2', () => {
    expect(isShardSeed('GLM-5.1-UD-Q4_K_M-00002-of-00011.gguf')).toBe(false)
  })

  it('returns false for a non-shard file', () => {
    expect(isShardSeed('model.gguf')).toBe(false)
  })
})

import { groupShardSets, buildShardMap, type ShardSet } from '../../src/inference/shard-utils'
import type { GGUFHeader, GGMLType, GGUFValueType } from '../../src/inference/types'

function makeShardHeader(tensorNames: string[], dataRegionOffset = 1024n): GGUFHeader {
  return {
    version: 3,
    tensorCount: BigInt(tensorNames.length),
    metadata: [{ key: 'general.architecture', valueType: 4 as GGUFValueType, value: 'llama' }],
    tensors: tensorNames.map((name, i) => ({
      name,
      shape: [4096n, 4096n],
      type: 12 as GGMLType,  // Q4_K
      dataOffset: BigInt(i * 1024 * 1024),
    })),
    metadataEndOffset: 100,
    dataRegionOffset,
  }
}

describe('groupShardSets', () => {
  it('groups multi-shard files into one ShardSet', () => {
    const files = [
      { rfilename: 'GLM-00001-of-00003.gguf', size: 100 },
      { rfilename: 'GLM-00002-of-00003.gguf', size: 200 },
      { rfilename: 'GLM-00003-of-00003.gguf', size: 150 },
    ]
    const result = groupShardSets(files)
    expect(result).toHaveLength(1)
    const set = result[0] as ShardSet
    expect(set.canonical).toBe('GLM-00001-of-00003.gguf')
    expect(set.shardFiles).toEqual([
      'GLM-00001-of-00003.gguf',
      'GLM-00002-of-00003.gguf',
      'GLM-00003-of-00003.gguf',
    ])
    expect(set.totalShards).toBe(3)
    expect(set.combinedSize).toBe(450)
  })

  it('passes non-shard files through unchanged', () => {
    const files = [
      { rfilename: 'model-Q4_K_M.gguf', size: 5000 },
    ]
    const result = groupShardSets(files)
    expect(result).toHaveLength(1)
    expect('rfilename' in result[0]).toBe(true)
  })

  it('handles mixed shard and non-shard files', () => {
    const files = [
      { rfilename: 'big-00001-of-00002.gguf', size: 100 },
      { rfilename: 'small-Q4_K_M.gguf', size: 50 },
      { rfilename: 'big-00002-of-00002.gguf', size: 100 },
    ]
    const result = groupShardSets(files)
    expect(result).toHaveLength(2)
  })
})

describe('buildShardMap', () => {
  it('maps tensor names to their shard filename and offset', () => {
    const shards = [
      {
        filename: 'model-00001-of-00002.gguf',
        header: makeShardHeader(['blk.0.attn_q.weight'], 512n),
      },
      {
        filename: 'model-00002-of-00002.gguf',
        header: makeShardHeader(['blk.1.attn_q.weight'], 512n),
      },
    ]
    const map = buildShardMap(shards)
    expect(map.get('blk.0.attn_q.weight')?.shardFilename).toBe('model-00001-of-00002.gguf')
    expect(map.get('blk.1.attn_q.weight')?.shardFilename).toBe('model-00002-of-00002.gguf')
  })
})
