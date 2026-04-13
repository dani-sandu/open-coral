import { app, dialog, ipcMain } from 'electron'
import { existsSync, statSync, openSync, readSync, closeSync, readFileSync, readdirSync } from 'fs'
import { join } from 'path'
import { parseGGUFHeader } from '../inference/gguf-parser'
import type { GGUFHeader } from '../inference/types'
import { countBlocks } from '../inference/block-extractor'

export interface ModelInfo {
  path: string
  architecture: string
  totalBlocks: number
  hiddenSize: number
  headCount: number
  fileSizeBytes: number
  /** First block's tensor suffixes (e.g. attn_q.weight, attn_qkv.weight) for compatibility checks */
  blockTensorSuffixes: string[]
  /** HuggingFace repo ID, e.g. "bartowski/Llama-3.2-3B-Instruct-GGUF". Present only for HF downloads. */
  repoId?: string
  /** HuggingFace filename, e.g. "Llama-3.2-3B-Instruct-Q4_K_M.gguf". Present only for HF downloads. */
  hfFilename?: string
}

let currentModel: ModelInfo | null = null
let currentGGUFHeader: GGUFHeader | null = null

/**
 * Parse GGUF metadata from a file path.
 * Returns both the ModelInfo summary and the raw GGUFHeader.
 * Reads only the first 80 MB — enough for the header of any model.
 */
function loadModelInfo(filePath: string): { info: ModelInfo; header: GGUFHeader } {
  const stat = statSync(filePath)
  const fd = openSync(filePath, 'r')
  // Read up to 80 MB — large models have tensor info tables that exceed 4 MB
  const readSize = Math.min(80 * 1024 * 1024, stat.size)
  const headerBuf = Buffer.allocUnsafe(readSize)
  readSync(fd, headerBuf, 0, headerBuf.length, 0)
  closeSync(fd)

  const header = parseGGUFHeader(headerBuf)

  const metaGet = (key: string): unknown =>
    header.metadata.find(m => m.key === key)?.value ?? null

  const architecture = (metaGet('general.architecture') as string | null) ?? 'unknown'
  const prefix = architecture === 'unknown' ? 'llama' : architecture
  const hiddenSize = Number(metaGet(`${prefix}.embedding_length`) ?? metaGet('llama.embedding_length') ?? 0)
  const headCount = Number(metaGet(`${prefix}.attention.head_count`) ?? metaGet('llama.attention.head_count') ?? 0)
  // Prefer the metadata block_count — always reflects the full model even in partial GGUF files.
  // countBlocks() only counts tensors present in the file, which is wrong for partial downloads.
  const blockCountFromMeta = Number(metaGet(`${prefix}.block_count`) ?? metaGet('llama.block_count') ?? 0)
  const totalBlocks = blockCountFromMeta > 0 ? blockCountFromMeta : countBlocks(header)

  // Extract tensor name suffixes for block 0 to identify architecture variant
  const blk0Prefix = 'blk.0.'
  const blockTensorSuffixes = header.tensors
    .filter(t => t.name.startsWith(blk0Prefix))
    .map(t => t.name.slice(blk0Prefix.length))

  // Read HuggingFace identity from sidecar if present
  let repoId: string | undefined
  let hfFilename: string | undefined
  const sidecarPath = filePath + '.opencoral-meta.json'
  try {
    const sidecar = JSON.parse(readFileSync(sidecarPath, 'utf-8'))
    repoId    = typeof sidecar.repoId    === 'string' ? sidecar.repoId    : undefined
    hfFilename = typeof sidecar.hfFilename === 'string' ? sidecar.hfFilename : undefined
  } catch (err: unknown) {
    if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err
    // ENOENT = no sidecar file — this is a local file with no HF identity
  }

  return {
    info: {
      path: filePath,
      architecture,
      totalBlocks,
      hiddenSize,
      headCount,
      fileSizeBytes: stat.size,
      blockTensorSuffixes,
      repoId,
      hfFilename,
    },
    header,
  }
}

function loadAndSet(filePath: string, repoId: string, hfFilename: string): ModelInfo {
  const loaded = loadModelInfo(filePath)
  currentModel = loaded.info
  currentGGUFHeader = loaded.header
  if (!currentModel.repoId) currentModel = { ...currentModel, repoId, hfFilename }
  return currentModel
}

export function setupModelIPC(): void {
  ipcMain.handle('opencoral:select-model', async (): Promise<ModelInfo | null> => {
    const result = await dialog.showOpenDialog({
      title: 'Select GGUF Model File',
      filters: [{ name: 'GGUF Models', extensions: ['gguf'] }],
      properties: ['openFile'],
    })
    if (result.canceled || result.filePaths.length === 0) return null

    const loaded = loadModelInfo(result.filePaths[0])
    currentModel = loaded.info
    currentGGUFHeader = loaded.header
    return currentModel
  })

  ipcMain.handle('opencoral:load-model-path', async (_event, filePath: string): Promise<ModelInfo> => {
    const loaded = loadModelInfo(filePath)
    currentModel = loaded.info
    currentGGUFHeader = loaded.header
    return currentModel
  })

  ipcMain.handle('opencoral:get-model', (): ModelInfo | null => currentModel)

  ipcMain.handle('opencoral:load-model-by-hf-identity', async (
    _event,
    repoId: string,
    hfFilename: string,
  ): Promise<ModelInfo> => {
    // Validate filename to prevent path traversal
    if (hfFilename.includes('/') || hfFilename.includes('\\') || hfFilename.includes('..')) {
      throw new Error('Invalid HF filename')
    }

    const modelsDir = join(app.getPath('userData'), 'models')

    // First, try the exact filename (full download)
    const exact = join(modelsDir, hfFilename)
    if (existsSync(exact)) {
      return loadAndSet(exact, repoId, hfFilename)
    }

    // Scan for matching .opencoral-meta.json sidecar (covers partial downloads with renamed files)
    const files = readdirSync(modelsDir).filter(f => f.endsWith('.opencoral-meta.json'))
    for (const sidecarFile of files) {
      const sidecarPath = join(modelsDir, sidecarFile)
      try {
        const meta = JSON.parse(readFileSync(sidecarPath, 'utf-8'))
        if (meta.repoId === repoId && meta.hfFilename === hfFilename) {
          // The GGUF file path = sidecar path minus '.opencoral-meta.json'
          const ggufPath = sidecarPath.slice(0, -'.opencoral-meta.json'.length)
          if (existsSync(ggufPath)) {
            return loadAndSet(ggufPath, repoId, hfFilename)
          }
        }
      } catch {
        // Ignore malformed sidecars
      }
    }

    throw new Error(`Model file not found locally: ${hfFilename}`)
  })

  ipcMain.handle('opencoral:list-local-models', (): LocalModelEntry[] => {
    return listLocalModels()
  })
}

export interface LocalModelEntry {
  path: string
  filename: string
  architecture: string
  totalBlocks: number
  hiddenSize: number
  headCount: number
  fileSizeBytes: number
  repoId?: string
  hfFilename?: string
  /** Block range present in the file (null = full model) */
  blockStart: number | null
  blockEnd: number | null
}

// Cache parsed model entries by file path + mtime to avoid re-reading 80MB headers on every call
const modelCache = new Map<string, { mtimeMs: number; entry: LocalModelEntry }>()

/**
 * Scan the userData/models directory and return metadata about every downloaded GGUF file.
 * Uses an mtime-based cache so unchanged files are not re-parsed.
 */
export function listLocalModels(): LocalModelEntry[] {
  const dir = join(app.getPath('userData'), 'models')
  if (!existsSync(dir)) return []

  const entries: LocalModelEntry[] = []
  const ggufFiles = readdirSync(dir).filter(f => f.endsWith('.gguf'))
  const seenPaths = new Set<string>()

  for (const file of ggufFiles) {
    const filePath = join(dir, file)
    seenPaths.add(filePath)
    try {
      const stat = statSync(filePath)

      // Check cache — skip parsing if file hasn't changed
      const cached = modelCache.get(filePath)
      if (cached && cached.mtimeMs === stat.mtimeMs) {
        entries.push(cached.entry)
        continue
      }

      const fd = openSync(filePath, 'r')
      const readSize = Math.min(80 * 1024 * 1024, stat.size)
      const headerBuf = Buffer.allocUnsafe(readSize)
      readSync(fd, headerBuf, 0, headerBuf.length, 0)
      closeSync(fd)

      const header = parseGGUFHeader(headerBuf)
      const metaGet = (key: string): unknown => header.metadata.find(m => m.key === key)?.value ?? null

      const architecture = (metaGet('general.architecture') as string | null) ?? 'unknown'
      const prefix = architecture === 'unknown' ? 'llama' : architecture
      const hiddenSize = Number(metaGet(`${prefix}.embedding_length`) ?? metaGet('llama.embedding_length') ?? 0)
      const headCount = Number(metaGet(`${prefix}.attention.head_count`) ?? metaGet('llama.attention.head_count') ?? 0)
      const blockCountFromMeta = Number(metaGet(`${prefix}.block_count`) ?? metaGet('llama.block_count') ?? 0)
      const totalBlocks = blockCountFromMeta > 0 ? blockCountFromMeta : countBlocks(header)

      let repoId: string | undefined
      let hfFilename: string | undefined
      let blockStart: number | null = null
      let blockEnd: number | null = null

      const sidecarPath = filePath + '.opencoral-meta.json'
      try {
        const sidecar = JSON.parse(readFileSync(sidecarPath, 'utf-8'))
        repoId     = typeof sidecar.repoId     === 'string' ? sidecar.repoId     : undefined
        hfFilename = typeof sidecar.hfFilename  === 'string' ? sidecar.hfFilename  : undefined
        blockStart = typeof sidecar.blockStart  === 'number' ? sidecar.blockStart  : null
        blockEnd   = typeof sidecar.blockEnd    === 'number' ? sidecar.blockEnd    : null
      } catch { /* ENOENT or parse error — no sidecar */ }

      const entry: LocalModelEntry = {
        path: filePath,
        filename: file,
        architecture,
        totalBlocks,
        hiddenSize,
        headCount,
        fileSizeBytes: stat.size,
        repoId,
        hfFilename,
        blockStart,
        blockEnd,
      }

      modelCache.set(filePath, { mtimeMs: stat.mtimeMs, entry })
      entries.push(entry)
    } catch {
      // Skip files that fail to parse
    }
  }

  // Evict stale cache entries
  for (const key of modelCache.keys()) {
    if (!seenPaths.has(key)) modelCache.delete(key)
  }

  return entries
}

export function getCurrentModel(): ModelInfo | null {
  return currentModel
}

export function getCurrentGGUFHeader(): GGUFHeader | null {
  return currentGGUFHeader
}
