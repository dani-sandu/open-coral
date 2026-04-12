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
  const sidecarPath = filePath + '.coral-meta.json'
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
  ipcMain.handle('coral:select-model', async (): Promise<ModelInfo | null> => {
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

  ipcMain.handle('coral:load-model-path', async (_event, filePath: string): Promise<ModelInfo> => {
    const loaded = loadModelInfo(filePath)
    currentModel = loaded.info
    currentGGUFHeader = loaded.header
    return currentModel
  })

  ipcMain.handle('coral:get-model', (): ModelInfo | null => currentModel)

  ipcMain.handle('coral:load-model-by-hf-identity', async (
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

    // Scan for matching .coral-meta.json sidecar (covers partial downloads with renamed files)
    const files = readdirSync(modelsDir).filter(f => f.endsWith('.coral-meta.json'))
    for (const sidecarFile of files) {
      const sidecarPath = join(modelsDir, sidecarFile)
      try {
        const meta = JSON.parse(readFileSync(sidecarPath, 'utf-8'))
        if (meta.repoId === repoId && meta.hfFilename === hfFilename) {
          // The GGUF file path = sidecar path minus '.coral-meta.json'
          const ggufPath = sidecarPath.slice(0, -'.coral-meta.json'.length)
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
}

export function getCurrentModel(): ModelInfo | null {
  return currentModel
}

export function getCurrentGGUFHeader(): GGUFHeader | null {
  return currentGGUFHeader
}
