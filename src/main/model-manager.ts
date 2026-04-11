import { dialog, ipcMain } from 'electron'
import { statSync, openSync, readSync, closeSync } from 'fs'
import { parseGGUFHeader } from '../inference/gguf-parser'
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
}

let currentModel: ModelInfo | null = null

/**
 * Parse GGUF metadata from a file path.
 * Reads only the first 4 MB — enough for the header of any model.
 */
function loadModelInfo(filePath: string): ModelInfo {
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
  const totalBlocks = countBlocks(header)

  // Extract tensor name suffixes for block 0 to identify architecture variant
  const blk0Prefix = 'blk.0.'
  const blockTensorSuffixes = header.tensors
    .filter(t => t.name.startsWith(blk0Prefix))
    .map(t => t.name.slice(blk0Prefix.length))

  return {
    path: filePath,
    architecture,
    totalBlocks,
    hiddenSize,
    headCount,
    fileSizeBytes: stat.size,
    blockTensorSuffixes,
  }
}

export function setupModelIPC(): void {
  ipcMain.handle('coral:select-model', async (): Promise<ModelInfo | null> => {
    const result = await dialog.showOpenDialog({
      title: 'Select GGUF Model File',
      filters: [{ name: 'GGUF Models', extensions: ['gguf'] }],
      properties: ['openFile'],
    })
    if (result.canceled || result.filePaths.length === 0) return null

    currentModel = loadModelInfo(result.filePaths[0])
    return currentModel
  })

  ipcMain.handle('coral:load-model-path', async (_event, filePath: string): Promise<ModelInfo> => {
    currentModel = loadModelInfo(filePath)
    return currentModel
  })

  ipcMain.handle('coral:get-model', (): ModelInfo | null => currentModel)
}

export function getCurrentModel(): ModelInfo | null {
  return currentModel
}
