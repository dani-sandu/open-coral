import { app, ipcMain } from 'electron'
import { createWriteStream, writeFileSync, mkdirSync, existsSync, statSync, unlinkSync } from 'fs'
import { join } from 'path'
import https from 'https'
import http from 'http'
import { parseGGUFHeader } from '../inference/gguf-parser'
import { countBlocks, extractBlockTensors, extractShimTensors } from '../inference/block-extractor'
import { tensorDataSize, tensorByteRanges, buildPartialGGUF, estimatePartialSize } from '../inference/gguf-partial'

// ── HF Hub API types ────────────────────────────────────────────────────────────

export interface HFModelResult {
  id: string              // e.g. "TheBloke/Llama-2-7B-GGUF"
  author: string
  modelId: string         // same as id
  likes: number
  downloads: number
  tags: string[]
  lastModified: string
}

export interface HFFileInfo {
  rfilename: string       // e.g. "llama-2-7b.Q4_K_M.gguf"
  size: number            // bytes
}

export interface DownloadProgress {
  file: string
  downloadedBytes: number
  totalBytes: number
  percent: number
  done: boolean
  error?: string
  localPath?: string
}

export interface HFModelPreview {
  architecture: string
  totalBlocks: number
  hiddenSize: number
  headCount: number
  repoId: string
  filename: string
}

// ── In-flight download tracking ─────────────────────────────────────────────────

let activeDownload: { abort: () => void } | null = null
let lastProgress: DownloadProgress | null = null

function modelsDir(): string {
  const dir = join(app.getPath('userData'), 'models')
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  return dir
}

// ── HF API helpers ──────────────────────────────────────────────────────────────

async function fetchJSON<T>(url: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const proto = url.startsWith('https') ? https : http
    proto.get(url, { headers: { 'User-Agent': 'OpenCoral/0.1' } }, (res) => {
      if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        return fetchJSON<T>(res.headers.location).then(resolve, reject)
      }
      if (res.statusCode !== 200) {
        return reject(new Error(`HF API ${res.statusCode}: ${url}`))
      }
      const chunks: Buffer[] = []
      res.on('data', (d: Buffer) => chunks.push(d))
      res.on('end', () => {
        try {
          resolve(JSON.parse(Buffer.concat(chunks).toString()))
        } catch (e) {
          reject(e)
        }
      })
      res.on('error', reject)
    }).on('error', reject)
  })
}

async function searchModels(query: string): Promise<HFModelResult[]> {
  const q = encodeURIComponent(query)
  // filter=gguf narrows to repos tagged with "gguf"
  const url = `https://huggingface.co/api/models?search=${q}&filter=gguf&sort=downloads&direction=-1&limit=20`
  const raw = await fetchJSON<any[]>(url)
  return raw.map(m => ({
    id: m.id ?? m.modelId ?? '',
    author: m.author ?? (m.id ?? '').split('/')[0] ?? '',
    modelId: m.id ?? m.modelId ?? '',
    likes: m.likes ?? 0,
    downloads: m.downloads ?? 0,
    tags: m.tags ?? [],
    lastModified: m.lastModified ?? '',
  }))
}

async function listGGUFFiles(repoId: string): Promise<HFFileInfo[]> {
  const url = `https://huggingface.co/api/models/${repoId}`
  const repo = await fetchJSON<any>(url)
  const siblings: any[] = repo.siblings ?? []
  return siblings
    .filter((s: any) => typeof s.rfilename === 'string' && s.rfilename.endsWith('.gguf'))
    .map((s: any) => ({ rfilename: s.rfilename, size: s.size ?? 0 }))
}

function downloadFile(
  repoId: string,
  filename: string,
  onProgress: (p: DownloadProgress) => void,
): { promise: Promise<string>; abort: () => void } {
  const localPath = join(modelsDir(), filename)
  let aborted = false
  let req: http.ClientRequest | null = null

  // If already fully downloaded, skip
  if (existsSync(localPath)) {
    try {
      const stat = statSync(localPath)
      if (stat.size > 0) {
        onProgress({ file: filename, downloadedBytes: stat.size, totalBytes: stat.size, percent: 100, done: true, localPath })
        return { promise: Promise.resolve(localPath), abort: () => {} }
      }
    } catch {}
  }

  const promise = new Promise<string>((resolve, reject) => {
    const url = `https://huggingface.co/${repoId}/resolve/main/${encodeURIComponent(filename)}`

    function follow(u: string): void {
      const proto = u.startsWith('https') ? https : http
      req = proto.get(u, { headers: { 'User-Agent': 'OpenCoral/0.1' } }, (res) => {
        if (aborted) return

        // Follow redirects (HF CDN uses 302)
        if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          return follow(res.headers.location)
        }

        if (res.statusCode !== 200) {
          return reject(new Error(`Download failed: HTTP ${res.statusCode}`))
        }

        const totalBytes = parseInt(res.headers['content-length'] ?? '0', 10)
        let downloadedBytes = 0

        const ws = createWriteStream(localPath)
        res.on('data', (chunk: Buffer) => {
          if (aborted) return
          downloadedBytes += chunk.length
          const percent = totalBytes > 0 ? Math.round((downloadedBytes / totalBytes) * 100) : 0
          onProgress({ file: filename, downloadedBytes, totalBytes, percent, done: false })
        })
        res.pipe(ws)
        ws.on('finish', () => {
          if (aborted) return
          onProgress({ file: filename, downloadedBytes, totalBytes, percent: 100, done: true, localPath })
          resolve(localPath)
        })
        ws.on('error', (err) => {
          try { unlinkSync(localPath) } catch {}
          reject(err)
        })
        res.on('error', (err) => {
          try { unlinkSync(localPath) } catch {}
          reject(err)
        })
      })
      req!.on('error', (err) => {
        if (!aborted) reject(err)
      })
    }

    follow(url)
  })

  return {
    promise,
    abort: () => {
      aborted = true
      req?.destroy()
      try { unlinkSync(localPath) } catch {}
    },
  }
}

// ── IPC handlers ────────────────────────────────────────────────────────────────

/**
 * Download a shim GGUF containing only embedding + output tensors.
 * Callable from both IPC handlers and directly from main-process code.
 */
export async function downloadShimGGUF(repoId: string, filename: string): Promise<string> {
  // Fetch header if not already cached
  if (!cachedHeader || !cachedHeaderBuf) {
    const sizes = [10 * 1024 * 1024, 30 * 1024 * 1024, 80 * 1024 * 1024]
    for (const size of sizes) {
      cachedHeaderBuf = await downloadRange(repoId, filename, 0, size - 1)
      try {
        cachedHeader = parseGGUFHeader(cachedHeaderBuf)
        break
      } catch (e) {
        if (size === sizes[sizes.length - 1]) throw e
      }
    }
    if (!cachedHeader || !cachedHeaderBuf) throw new Error('Failed to parse GGUF header')
  }

  if (activeDownload) {
    activeDownload.abort()
    activeDownload = null
  }

  const header = cachedHeader
  const headerBuf = cachedHeaderBuf

  const { embeddingTensor, outputTensors } = extractShimTensors(header)
  const selectedTensors = [
    ...(embeddingTensor ? [embeddingTensor] : []),
    ...outputTensors,
  ]

  if (selectedTensors.length === 0) {
    throw new Error('No embedding or output tensors found in model')
  }

  const ranges = tensorByteRanges(header, selectedTensors)
  const totalDownload = ranges.reduce((s, r) => s + r.size, 0)
  let downloaded = 0
  let aborted = false

  const baseName = filename.replace(/\.gguf$/i, '')
  const shimName = `${baseName}.shim.gguf`
  const localPath = join(modelsDir(), shimName)

  if (existsSync(localPath)) {
    try { unlinkSync(localPath) } catch {}
  }

  lastProgress = { file: shimName, downloadedBytes: 0, totalBytes: totalDownload, percent: 0, done: false }

  const abortController = { abort: () => { aborted = true } }
  activeDownload = abortController

  try {
    const tensorDataChunks = new Map<string, Buffer>()
    for (const range of ranges) {
      if (aborted) throw new Error('Cancelled')
      const data = await downloadRange(repoId, filename, range.absoluteOffset, range.absoluteOffset + range.size - 1)
      tensorDataChunks.set(range.name, data)
      downloaded += data.length
      const percent = totalDownload > 0 ? Math.round((downloaded / totalDownload) * 100) : 0
      lastProgress = { file: shimName, downloadedBytes: downloaded, totalBytes: totalDownload, percent, done: false }
    }

    if (aborted) throw new Error('Cancelled')

    const partialBuf = buildPartialGGUF(headerBuf, header, selectedTensors, tensorDataChunks)
    writeFileSync(localPath, partialBuf)

    const meta = { repoId, hfFilename: filename, blockStart: null, blockEnd: null, shim: true }
    writeFileSync(localPath + '.opencoral-meta.json', JSON.stringify(meta, null, 2), 'utf-8')

    lastProgress = { file: shimName, downloadedBytes: downloaded, totalBytes: totalDownload, percent: 100, done: true, localPath }
    activeDownload = null
    return localPath
  } catch (err) {
    activeDownload = null
    try { unlinkSync(localPath) } catch {}
    throw err
  }
}

export function setupHuggingFaceIPC(): void {
  ipcMain.handle('opencoral:hf-search', async (_event, query: string): Promise<HFModelResult[]> => {
    if (!query || query.trim().length < 2) return []
    return searchModels(query.trim())
  })

  ipcMain.handle('opencoral:hf-list-files', async (_event, repoId: string): Promise<HFFileInfo[]> => {
    return listGGUFFiles(repoId)
  })

  ipcMain.handle('opencoral:hf-download', async (
    _event,
    repoId: string,
    filename: string,
  ): Promise<string> => {
    // Cancel any in-flight download
    if (activeDownload) {
      activeDownload.abort()
      activeDownload = null
    }

    const { promise, abort } = downloadFile(repoId, filename, (p) => {
      lastProgress = p
    })
    activeDownload = { abort }

    try {
      const localPath = await promise
      activeDownload = null
      // Write HF identity sidecar
      const fullMeta = { repoId, hfFilename: filename, blockStart: null, blockEnd: null }
      writeFileSync(localPath + '.opencoral-meta.json', JSON.stringify(fullMeta, null, 2), 'utf-8')
      return localPath
    } catch (err) {
      activeDownload = null
      throw err
    }
  })

  ipcMain.handle('opencoral:hf-download-progress', (): DownloadProgress | null => {
    return lastProgress
  })

  ipcMain.handle('opencoral:hf-cancel-download', (): void => {
    if (activeDownload) {
      activeDownload.abort()
      activeDownload = null
      if (lastProgress) {
        lastProgress = { ...lastProgress, done: true, error: 'Cancelled' }
      }
    }
  })

  // ── Partial download: header preview ─────────────────────────────────────

  ipcMain.handle('opencoral:hf-preview-model', async (
    _event,
    repoId: string,
    filename: string,
  ): Promise<HFModelPreview> => {
    // Try progressively larger chunks — tensor info tables can exceed 4 MB for large models
    const sizes = [10 * 1024 * 1024, 30 * 1024 * 1024, 80 * 1024 * 1024]
    let headerBuf: Buffer | null = null
    let header: ReturnType<typeof parseGGUFHeader> | null = null
    for (const size of sizes) {
      headerBuf = await downloadRange(repoId, filename, 0, size - 1)
      try {
        header = parseGGUFHeader(headerBuf)
        break
      } catch (e) {
        // If last attempt, rethrow
        if (size === sizes[sizes.length - 1]) throw e
      }
    }
    if (!header || !headerBuf) throw new Error('Failed to parse GGUF header')

    const metaGet = (key: string): unknown =>
      header.metadata.find(m => m.key === key)?.value ?? null
    const architecture = (metaGet('general.architecture') as string | null) ?? 'unknown'
    const prefix = architecture === 'unknown' ? 'llama' : architecture
    const hiddenSize = Number(metaGet(`${prefix}.embedding_length`) ?? metaGet('llama.embedding_length') ?? 0)
    const headCount = Number(metaGet(`${prefix}.attention.head_count`) ?? metaGet('llama.attention.head_count') ?? 0)
    const totalBlocks = countBlocks(header)

    // Cache the header for partial download
    cachedHeaderBuf = headerBuf
    cachedHeader = header

    return {
      architecture, totalBlocks, hiddenSize, headCount,
      repoId, filename,
    }
  })

  // ── Partial download: download only selected blocks ──────────────────────

  ipcMain.handle('opencoral:hf-estimate-blocks', async (
    _event,
    blockStart: number,
    blockEnd: number,
  ): Promise<{ partialSize: number; fullSize: number; savedPercent: number }> => {
    if (!cachedHeader) throw new Error('Call opencoral:hf-preview-model first')
    const total = countBlocks(cachedHeader)
    const extracted = extractBlockTensors(cachedHeader, { start: blockStart, end: blockEnd }, total)
    const allTensors = [
      ...extracted.blockTensors,
      ...(extracted.embeddingTensor ? [extracted.embeddingTensor] : []),
      ...extracted.outputTensors,
    ]
    const partialSize = estimatePartialSize(allTensors)
    const fullSize = Number(cachedHeader.tensors.reduce((s, t) => {
      return s + tensorDataSize(t.shape, t.type)
    }, 0)) + 4 * 1024 * 1024
    const savedPercent = fullSize > 0 ? Math.round((1 - partialSize / fullSize) * 100) : 0
    return { partialSize, fullSize, savedPercent }
  })

  ipcMain.handle('opencoral:hf-download-partial', async (
    _event,
    repoId: string,
    filename: string,
    blockStart: number,
    blockEnd: number,
  ): Promise<string> => {
    if (!cachedHeader || !cachedHeaderBuf) throw new Error('Call opencoral:hf-preview-model first')

    if (activeDownload) {
      activeDownload.abort()
      activeDownload = null
    }

    const header = cachedHeader
    const headerBuf = cachedHeaderBuf
    const total = countBlocks(header)
    const extracted = extractBlockTensors(header, { start: blockStart, end: blockEnd }, total)
    const selectedTensors = [
      ...(extracted.embeddingTensor ? [extracted.embeddingTensor] : []),
      ...extracted.blockTensors,
      ...extracted.outputTensors,
    ]

    const ranges = tensorByteRanges(header, selectedTensors)
    const totalDownload = ranges.reduce((s, r) => s + r.size, 0)
    let downloaded = 0
    let aborted = false

    const baseName = filename.replace(/\.gguf$/i, '')
    const partialName = `${baseName}.blocks-${blockStart}-${blockEnd}.gguf`
    const localPath = join(modelsDir(), partialName)

    // Always rebuild partial files to ensure format correctness
    if (existsSync(localPath)) {
      try { unlinkSync(localPath) } catch {}
    }

    lastProgress = { file: partialName, downloadedBytes: 0, totalBytes: totalDownload, percent: 0, done: false }

    const abortController = { abort: () => { aborted = true } }
    activeDownload = abortController

    try {
      // Download each tensor's byte range
      const tensorDataChunks = new Map<string, Buffer>()
      for (const range of ranges) {
        if (aborted) throw new Error('Cancelled')
        const data = await downloadRange(repoId, filename, range.absoluteOffset, range.absoluteOffset + range.size - 1)
        tensorDataChunks.set(range.name, data)
        downloaded += data.length
        const percent = totalDownload > 0 ? Math.round((downloaded / totalDownload) * 100) : 0
        lastProgress = { file: partialName, downloadedBytes: downloaded, totalBytes: totalDownload, percent, done: false }
      }

      if (aborted) throw new Error('Cancelled')

      // Build partial GGUF
      const partialBuf = buildPartialGGUF(headerBuf, header, selectedTensors, tensorDataChunks)
      writeFileSync(localPath, partialBuf)

      // Write HF identity sidecar
      const meta = { repoId, hfFilename: filename, blockStart, blockEnd }
      writeFileSync(localPath + '.opencoral-meta.json', JSON.stringify(meta, null, 2), 'utf-8')

      lastProgress = { file: partialName, downloadedBytes: downloaded, totalBytes: totalDownload, percent: 100, done: true, localPath }
      activeDownload = null
      return localPath
    } catch (err) {
      activeDownload = null
      try { unlinkSync(localPath) } catch {}
      throw err
    }
  })

  // ── Shim download: embed + output tensors only (no blocks) ────────────────

  ipcMain.handle('opencoral:hf-download-shim', async (
    _event,
    repoId: string,
    filename: string,
  ): Promise<string> => {
    return downloadShimGGUF(repoId, filename)
  })
}

// ── Cached header for partial download flow ───────────────────────────────────

let cachedHeaderBuf: Buffer | null = null
let cachedHeader: ReturnType<typeof parseGGUFHeader> | null = null

// ── HTTP Range request helper ─────────────────────────────────────────────────

function downloadRange(
  repoId: string,
  filename: string,
  rangeStart: number,
  rangeEnd: number,
): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const url = `https://huggingface.co/${repoId}/resolve/main/${encodeURIComponent(filename)}`

    function follow(u: string): void {
      const proto = u.startsWith('https') ? https : http
      proto.get(u, {
        headers: {
          'User-Agent': 'OpenCoral/0.1',
          'Range': `bytes=${rangeStart}-${rangeEnd}`,
        },
      }, (res) => {
        if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          return follow(res.headers.location)
        }
        if (res.statusCode !== 206 && res.statusCode !== 200) {
          return reject(new Error(`Range download failed: HTTP ${res.statusCode}`))
        }
        const chunks: Buffer[] = []
        res.on('data', (d: Buffer) => chunks.push(d))
        res.on('end', () => resolve(Buffer.concat(chunks)))
        res.on('error', reject)
      }).on('error', reject)
    }

    follow(url)
  })
}
