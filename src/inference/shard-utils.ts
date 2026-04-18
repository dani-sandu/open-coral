import type { GGUFHeader } from './types'
import { tensorDataSize } from './gguf-partial'

export interface ShardInfo {
  prefix: string       // base name without shard suffix, e.g. "GLM-5.1-UD-Q4_K_M"
  shardNo: number      // 1-based shard index
  totalShards: number
}

// Matches filenames like: SomeName-00003-of-00011.gguf
const SHARD_RE = /^(.+)-(\d{5})-of-(\d{5})\.gguf$/i

export function parseShardInfo(filename: string): ShardInfo | null {
  const m = SHARD_RE.exec(filename)
  if (!m) return null
  return {
    prefix: m[1],
    shardNo: parseInt(m[2], 10),
    totalShards: parseInt(m[3], 10),
  }
}

/** True only for the first shard (the one containing metadata). */
export function isShardSeed(filename: string): boolean {
  const info = parseShardInfo(filename)
  return info !== null && info.shardNo === 1
}

// NOTE: Keep in sync with ShardSet in src/renderer/src/types.ts.
// Duplicated intentionally — the renderer cannot import main-process modules.
export interface ShardSet {
  canonical: string      // shard 1 filename — used as identity across the network
  shardFiles: string[]   // all filenames sorted by shard number
  totalShards: number
  combinedSize: number   // sum of all shard sizes in bytes
}

/** Structural minimum for entries accepted by groupShardSets (matches HFFileInfo). */
export interface FileEntry {
  rfilename: string
  size: number
}

/** True if a value is a ShardSet (vs a plain FileEntry / HFFileInfo). */
export function isShardSet(f: FileEntry | ShardSet): f is ShardSet {
  return 'canonical' in f
}

/**
 * Group a flat list of GGUF file entries into ShardSets (multi-shard) and
 * pass-through entries (single-file). Order within each ShardSet is by shard number.
 */
export function groupShardSets(files: FileEntry[]): (ShardSet | FileEntry)[] {
  const groups = new Map<string, FileEntry[]>()
  const nonShards: FileEntry[] = []

  for (const f of files) {
    const info = parseShardInfo(f.rfilename)
    if (info) {
      const key = `${info.prefix}::${info.totalShards}`
      if (!groups.has(key)) groups.set(key, [])
      groups.get(key)!.push(f)
    } else {
      nonShards.push(f)
    }
  }

  const result: (ShardSet | FileEntry)[] = [...nonShards]

  for (const shards of groups.values()) {
    shards.sort((a, b) => parseShardInfo(a.rfilename)!.shardNo - parseShardInfo(b.rfilename)!.shardNo)
    result.push({
      canonical: shards[0].rfilename,
      shardFiles: shards.map(s => s.rfilename),
      totalShards: parseShardInfo(shards[0].rfilename)!.totalShards,
      combinedSize: shards.reduce((sum, s) => sum + (s.size ?? 0), 0),
    })
  }

  return result
}

/** Per-tensor location record used during partial cross-shard downloads. */
export interface ShardTensorLocation {
  shardFilename: string
  absoluteByteOffset: number
  size: number
}

/** Maps tensor name → its location within a specific shard file. */
export type ShardMap = Map<string, ShardTensorLocation>

/**
 * Build a ShardMap from a list of (filename, parsed-header) pairs.
 * Each header should be from parseGGUFHeader() called on a single shard.
 */
export function buildShardMap(
  shards: Array<{ filename: string; header: GGUFHeader }>
): ShardMap {
  const map: ShardMap = new Map()
  for (const { filename, header } of shards) {
    const dataStart = Number(header.dataRegionOffset)
    for (const t of header.tensors) {
      map.set(t.name, {
        shardFilename: filename,
        absoluteByteOffset: dataStart + Number(t.dataOffset),
        size: tensorDataSize(t.shape, t.type),
      })
    }
  }
  return map
}
