import React, { useState, useCallback, useEffect, useRef } from 'react'
import type { ModelInfo, HFModelResult, HFFileInfo, HFModelPreview, DownloadProgress, BlockEstimate } from './types'

const C = {
  bg: '#1e1e2e', surface: '#181825', border: '#313244',
  text: '#cdd6f4', dim: '#6c7086', accent: '#7c6af7', accentDim: '#45396e',
  red: '#f38ba8', green: '#a6e3a1', yellow: '#f9e2af', blue: '#89b4fa',
}

function fmt(bytes: number): string {
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(1) + ' GB'
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + ' MB'
  return (bytes / 1e3).toFixed(0) + ' KB'
}

function fmtCount(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M'
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K'
  return String(n)
}

type View = 'home' | 'search' | 'files' | 'preview' | 'downloading'

// ── Quantization ranking & helpers ──────────────────────────────────────────

interface QuantInfo {
  quant: string
  quality: 'low' | 'medium' | 'high'
  label: string
  order: number
}

const QUANT_TABLE: Record<string, QuantInfo> = {
  'Q2_K':     { quant: 'Q2_K',     quality: 'low',    label: 'Smallest — fast, lower quality', order: 1 },
  'Q3_K_S':   { quant: 'Q3_K_S',   quality: 'low',    label: 'Small — fast, lower quality', order: 2 },
  'Q3_K_M':   { quant: 'Q3_K_M',   quality: 'low',    label: 'Small — balanced for low RAM', order: 3 },
  'Q3_K_L':   { quant: 'Q3_K_L',   quality: 'medium', label: 'Small-medium', order: 4 },
  'IQ4_XS':   { quant: 'IQ4_XS',   quality: 'medium', label: 'Compact medium quality', order: 5 },
  'Q4_0':     { quant: 'Q4_0',     quality: 'medium', label: 'Medium — legacy quantization', order: 6 },
  'Q4_K_S':   { quant: 'Q4_K_S',   quality: 'medium', label: 'Medium — good speed', order: 7 },
  'Q4_K_M':   { quant: 'Q4_K_M',   quality: 'medium', label: 'Recommended — best balance of size & quality', order: 8 },
  'Q5_0':     { quant: 'Q5_0',     quality: 'high',   label: 'High — legacy quantization', order: 9 },
  'Q5_K_S':   { quant: 'Q5_K_S',   quality: 'high',   label: 'High quality', order: 10 },
  'Q5_K_M':   { quant: 'Q5_K_M',   quality: 'high',   label: 'High quality — slightly larger', order: 11 },
  'Q6_K':     { quant: 'Q6_K',     quality: 'high',   label: 'Very high quality — large', order: 12 },
  'Q8_0':     { quant: 'Q8_0',     quality: 'high',   label: 'Near-lossless — very large', order: 13 },
  'F16':      { quant: 'F16',      quality: 'high',   label: 'Full precision fp16 — largest', order: 14 },
  'F32':      { quant: 'F32',      quality: 'high',   label: 'Full precision fp32 — largest', order: 15 },
  'BF16':     { quant: 'BF16',     quality: 'high',   label: 'Full precision bf16 — largest', order: 16 },
}

const RECOMMENDED_QUANT = 'Q4_K_M'

function extractQuant(filename: string): string | null {
  // Match patterns like Q4_K_M, Q8_0, IQ4_XS, F16, BF16 in the filename
  const m = filename.match(/[.-]((?:IQ|Q|F|BF)\d+(?:_K)?(?:_[A-Z]+)?|F16|F32|BF16)(?:[.-]|$)/i)
  return m ? m[1].toUpperCase() : null
}

function getQuantInfo(quant: string): QuantInfo {
  return QUANT_TABLE[quant] ?? { quant, quality: 'medium' as const, label: quant, order: 50 }
}

const qualityColor: Record<string, string> = {
  low: C.yellow,
  medium: C.green,
  high: C.blue,
}


export default function ModelPanel(): React.JSX.Element {
  const [model, setModel] = useState<ModelInfo | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  // HF search state
  const [view, setView] = useState<View>('home')
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<HFModelResult[]>([])
  const [searching, setSearching] = useState(false)

  // HF file list state
  const [selectedRepo, setSelectedRepo] = useState<HFModelResult | null>(null)
  const [files, setFiles] = useState<HFFileInfo[]>([])
  const [loadingFiles, setLoadingFiles] = useState(false)
  const [showAllFiles, setShowAllFiles] = useState(false)

  // Download state
  const [downloading, setDownloading] = useState(false)
  const [progress, setProgress] = useState<DownloadProgress | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Preview state (block picker before download)
  const [preview, setPreview] = useState<HFModelPreview | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [selectedFile, setSelectedFile] = useState<HFFileInfo | null>(null)
  const [pBlockStart, setPBlockStart] = useState(0)
  const [pBlockEnd, setPBlockEnd] = useState(0)
  const [estimate, setEstimate] = useState<BlockEstimate | null>(null)

  // Load existing model on mount
  useEffect(() => {
    window.opencoral.getModel().then(m => { if (m) setModel(m) })
  }, [])

  // Poll download progress
  useEffect(() => {
    if (!downloading) return
    pollRef.current = setInterval(async () => {
      const p = await window.opencoral.hfDownloadProgress()
      if (p) setProgress(p)
      if (p?.done) {
        setDownloading(false)
        if (p.localPath && !p.error) {
          try {
            const m = await window.opencoral.loadModelPath(p.localPath)
            setModel(m)
            setView('home')
          } catch (e) {
            setError(String(e))
            setView('home')
          }
        } else if (p.error) {
          setError(p.error)
          setView('files')
        }
      }
    }, 300)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [downloading])

  const pick = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const m = await window.opencoral.selectModel()
      if (m) setModel(m)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  const search = useCallback(async () => {
    if (!query.trim()) return
    setSearching(true)
    setError(null)
    try {
      const r = await window.opencoral.hfSearch(query.trim())
      setResults(r)
      setView('search')
    } catch (e) {
      setError(String(e))
    } finally {
      setSearching(false)
    }
  }, [query])

  const selectRepo = useCallback(async (repo: HFModelResult) => {
    setSelectedRepo(repo)
    setLoadingFiles(true)
    setError(null)
    setShowAllFiles(false)
    try {
      const f = await window.opencoral.hfListFiles(repo.id)
      // Sort by quantization quality rank, recommended first
      f.sort((a, b) => {
        const qa = extractQuant(a.rfilename)
        const qb = extractQuant(b.rfilename)
        const oa = qa ? getQuantInfo(qa).order : 50
        const ob = qb ? getQuantInfo(qb).order : 50
        return oa - ob
      })
      setFiles(f)
      setView('files')
    } catch (e) {
      setError(String(e))
    } finally {
      setLoadingFiles(false)
    }
  }, [])

  const cancelDownload = useCallback(async () => {
    await window.opencoral.hfCancelDownload()
    setDownloading(false)
    setView('files')
  }, [])

  const selectFileForPreview = useCallback(async (file: HFFileInfo) => {
    if (!selectedRepo) return
    setSelectedFile(file)
    setPreviewLoading(true)
    setPreview(null)
    setEstimate(null)
    setError(null)
    setView('preview')
    try {
      const p = await window.opencoral.hfPreviewModel(selectedRepo.id, file.rfilename)
      setPreview(p)
      setPBlockStart(0)
      setPBlockEnd(Math.min(3, p.totalBlocks - 1))
      // Auto-estimate for initial range
      const est = await window.opencoral.hfEstimateBlocks(0, Math.min(3, p.totalBlocks - 1))
      setEstimate(est)
    } catch (e) {
      setError(String(e))
      setView('files')
    } finally {
      setPreviewLoading(false)
    }
  }, [selectedRepo])

  const updateEstimate = useCallback(async (start: number, end: number) => {
    setPBlockStart(start)
    setPBlockEnd(end)
    try {
      const est = await window.opencoral.hfEstimateBlocks(start, end)
      setEstimate(est)
    } catch {
      setEstimate(null)
    }
  }, [])

  const startPartialDownload = useCallback(async (full = false) => {
    if (!selectedRepo || !selectedFile) return
    setDownloading(true)
    setProgress(null)
    setError(null)
    setView('downloading')
    try {
      if (full) {
        await window.opencoral.hfDownload(selectedRepo.id, selectedFile.rfilename)
      } else {
        await window.opencoral.hfDownloadPartial(selectedRepo.id, selectedFile.rfilename, pBlockStart, pBlockEnd)
      }
    } catch (e) {
      setError(String(e))
      setDownloading(false)
      setView('preview')
    }
  }, [selectedRepo, selectedFile, pBlockStart, pBlockEnd])

  return (
    <div style={{ padding: 20, fontFamily: 'system-ui' }}>
      <h2 style={{ color: C.text, fontSize: 16, margin: '0 0 16px' }}>
        <span style={{ color: C.accent }}>⬡</span> Model
      </h2>

      {error && (
        <div style={{ color: C.red, fontSize: 12, marginBottom: 12, padding: '6px 10px', background: C.red + '11', borderRadius: 6 }}>{error}</div>
      )}

      {/* ── Loaded model card ─────────────────────────────────────── */}
      {model && view === 'home' && (
        <div style={{
          background: C.surface, border: `1px solid ${C.border}`,
          borderRadius: 10, padding: 16, fontFamily: 'monospace', fontSize: 12, marginBottom: 20,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
            <span style={{ color: C.accent, fontWeight: 600, fontSize: 13 }}>
              {model.path.split(/[\\/]/).pop()}
            </span>
            <span style={{ color: C.green, fontSize: 11 }}>Loaded</span>
          </div>
          {([
            ['Architecture', model.architecture],
            ['Total blocks', model.totalBlocks],
            ['Hidden size', model.hiddenSize],
            ['Heads', model.headCount],
            ['File size', fmt(model.fileSizeBytes)],
          ] as [string, string | number][]).map(([label, value]) => (
            <div key={label} style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', borderBottom: `1px solid ${C.border}` }}>
              <span style={{ color: C.dim }}>{label}</span>
              <span style={{ color: C.text }}>{value}</span>
            </div>
          ))}
        </div>
      )}

      {/* ── Home: pick source ─────────────────────────────────────── */}
      {view === 'home' && (
        <div>
          <div style={{ fontSize: 12, color: C.dim, marginBottom: 14 }}>
            {model ? 'Load a different model:' : 'Choose a model source:'}
          </div>

          {/* HF Search bar */}
          <div style={{
            display: 'flex', gap: 8, marginBottom: 12,
          }}>
            <input
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') search() }}
              placeholder="Search Hugging Face GGUF models…"
              style={{
                flex: 1, background: C.surface, color: C.text,
                border: `1px solid ${C.border}`, borderRadius: 8,
                padding: '10px 14px', fontSize: 13, outline: 'none',
              }}
            />
            <button
              onClick={search}
              disabled={searching || !query.trim()}
              style={{
                background: C.accent, color: '#fff', border: 'none',
                borderRadius: 8, padding: '10px 18px', fontSize: 13,
                cursor: searching ? 'default' : 'pointer',
                opacity: searching || !query.trim() ? 0.6 : 1,
              }}
            >
              {searching ? 'Searching…' : 'Search HF'}
            </button>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
            <div style={{ flex: 1, height: 1, background: C.border }} />
            <span style={{ color: C.dim, fontSize: 11 }}>or</span>
            <div style={{ flex: 1, height: 1, background: C.border }} />
          </div>

          <button
            onClick={pick}
            disabled={loading}
            style={{
              background: 'transparent', color: C.dim, border: `1px solid ${C.border}`,
              borderRadius: 8, padding: '10px 20px', fontSize: 13, width: '100%',
              cursor: loading ? 'default' : 'pointer', opacity: loading ? 0.6 : 1,
            }}
          >
            {loading ? 'Loading…' : 'Select local GGUF file'}
          </button>
        </div>
      )}

      {/* ── Search results ────────────────────────────────────────── */}
      {view === 'search' && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <button onClick={() => setView('home')} style={backBtnStyle}>Back</button>
            <span style={{ color: C.dim, fontSize: 11 }}>{results.length} results</span>
          </div>

          {results.length === 0 && (
            <div style={{ color: C.dim, fontSize: 12, padding: 20, textAlign: 'center' }}>
              No GGUF models found for "{query}"
            </div>
          )}

          {results.map(repo => (
            <button
              key={repo.id}
              onClick={() => selectRepo(repo)}
              style={{
                display: 'block', width: '100%', textAlign: 'left',
                background: C.surface, border: `1px solid ${C.border}`,
                borderRadius: 8, padding: '12px 14px', marginBottom: 6, cursor: 'pointer',
                transition: 'border-color 0.15s',
              }}
              onMouseEnter={e => (e.currentTarget.style.borderColor = C.accent + '66')}
              onMouseLeave={e => (e.currentTarget.style.borderColor = C.border)}
            >
              <div style={{ color: C.text, fontSize: 13, fontWeight: 600, marginBottom: 4 }}>
                {repo.id}
              </div>
              <div style={{ display: 'flex', gap: 14, fontSize: 11, color: C.dim }}>
                <span>Downloads: {fmtCount(repo.downloads)}</span>
                <span>Likes: {fmtCount(repo.likes)}</span>
              </div>
            </button>
          ))}
        </div>
      )}

      {/* ── File picker ───────────────────────────────────────────── */}
      {view === 'files' && selectedRepo && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <button onClick={() => setView('search')} style={backBtnStyle}>Back</button>
            <span style={{ color: C.accent, fontSize: 12, fontWeight: 600 }}>{selectedRepo.id}</span>
          </div>

          {loadingFiles && (
            <div style={{ color: C.dim, fontSize: 12, padding: 20, textAlign: 'center' }}>Loading files…</div>
          )}

          {!loadingFiles && files.length === 0 && (
            <div style={{ color: C.dim, fontSize: 12, padding: 20, textAlign: 'center' }}>
              No .gguf files found in this repository.
            </div>
          )}

          {!loadingFiles && files.length > 0 && (() => {
            // Find the recommended file
            const recommended = files.find(f => {
              const q = extractQuant(f.rfilename)
              return q === RECOMMENDED_QUANT
            })
            // Filter: show recommended + a few key quants, or all
            const keyQuants = new Set(['Q2_K', 'Q4_K_M', 'Q5_K_M', 'Q8_0'])
            const filteredFiles = showAllFiles
              ? files
              : files.filter(f => {
                  const q = extractQuant(f.rfilename)
                  return q ? keyQuants.has(q) : false
                })
            // If filtering left <2 files, show all
            const displayFiles = filteredFiles.length >= 2 ? filteredFiles : files
            const isFiltered = displayFiles.length < files.length

            return (
              <div>
                {/* Auto-recommended card */}
                {recommended && !showAllFiles && (
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ fontSize: 12, color: C.dim, marginBottom: 6 }}>
                      Recommended for you:
                    </div>
                    <button
                      onClick={() => selectFileForPreview(recommended)}
                      style={{
                        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                        width: '100%', textAlign: 'left',
                        background: C.accent + '11', border: `1px solid ${C.accent}44`,
                        borderRadius: 10, padding: '14px 16px', cursor: 'pointer',
                        transition: 'border-color 0.15s',
                      }}
                      onMouseEnter={e => (e.currentTarget.style.borderColor = C.accent)}
                      onMouseLeave={e => (e.currentTarget.style.borderColor = C.accent + '44')}
                    >
                      <div>
                        <div style={{ color: C.accent, fontSize: 13, fontWeight: 600, fontFamily: 'monospace', marginBottom: 4 }}>
                          {recommended.rfilename}
                        </div>
                        <div style={{ fontSize: 11, color: C.dim }}>
                          {QUANT_TABLE[RECOMMENDED_QUANT]?.label ?? 'Best balance of size & quality'}
                        </div>
                      </div>
                      <div style={{ textAlign: 'right', flexShrink: 0, marginLeft: 12 }}>
                        <div style={{ color: C.text, fontSize: 12 }}>
                          {recommended.size > 0 ? fmt(recommended.size) : ''}
                        </div>
                        <div style={{ color: C.accent, fontSize: 10, marginTop: 2 }}>Select</div>
                      </div>
                    </button>
                  </div>
                )}

                {/* Other variants */}
                <div style={{ fontSize: 12, color: C.dim, marginBottom: 6 }}>
                  {showAllFiles ? `All variants (${files.length}):` : 'Other variants:'}
                </div>

                {displayFiles.map(f => {
                  const quant = extractQuant(f.rfilename)
                  const info = quant ? getQuantInfo(quant) : null
                  const isRecommended = quant === RECOMMENDED_QUANT
                  if (!showAllFiles && isRecommended && recommended) return null

                  return (
                    <button
                      key={f.rfilename}
                      onClick={() => selectFileForPreview(f)}
                      style={{
                        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                        width: '100%', textAlign: 'left',
                        background: C.surface, border: `1px solid ${C.border}`,
                        borderRadius: 8, padding: '10px 14px', marginBottom: 4, cursor: 'pointer',
                        transition: 'border-color 0.15s',
                      }}
                      onMouseEnter={e => (e.currentTarget.style.borderColor = C.accent + '66')}
                      onMouseLeave={e => (e.currentTarget.style.borderColor = C.border)}
                    >
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2 }}>
                          <span style={{ color: C.text, fontSize: 12, fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {f.rfilename}
                          </span>
                          {info && (
                            <span style={{
                              fontSize: 9, padding: '1px 5px', borderRadius: 3,
                              background: qualityColor[info.quality] + '18',
                              color: qualityColor[info.quality],
                              border: `1px solid ${qualityColor[info.quality]}33`,
                              flexShrink: 0,
                            }}>
                              {info.quality}
                            </span>
                          )}
                        </div>
                        {info && (
                          <div style={{ fontSize: 10, color: C.dim }}>{info.label}</div>
                        )}
                      </div>
                      <span style={{ color: C.dim, fontSize: 11, flexShrink: 0, marginLeft: 12 }}>
                        {f.size > 0 ? fmt(f.size) : ''}
                      </span>
                    </button>
                  )
                })}

                {/* Show more / less toggle */}
                {isFiltered && (
                  <button
                    onClick={() => setShowAllFiles(true)}
                    style={{
                      display: 'block', width: '100%', marginTop: 8,
                      background: 'transparent', color: C.dim,
                      border: `1px dashed ${C.border}`, borderRadius: 8,
                      padding: '8px', fontSize: 11, cursor: 'pointer', textAlign: 'center',
                    }}
                  >
                    Show all {files.length} variants
                  </button>
                )}
                {showAllFiles && files.length > 4 && (
                  <button
                    onClick={() => setShowAllFiles(false)}
                    style={{
                      display: 'block', width: '100%', marginTop: 8,
                      background: 'transparent', color: C.dim,
                      border: `1px dashed ${C.border}`, borderRadius: 8,
                      padding: '8px', fontSize: 11, cursor: 'pointer', textAlign: 'center',
                    }}
                  >
                    Show fewer
                  </button>
                )}
              </div>
            )
          })()}
        </div>
      )}

      {/* ── Preview: block picker before download ─────────────────── */}
      {view === 'preview' && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <button onClick={() => setView('files')} style={backBtnStyle}>Back</button>
            <span style={{ color: C.accent, fontSize: 12, fontWeight: 600 }}>{selectedRepo?.id}</span>
          </div>

          {previewLoading && (
            <div style={{ color: C.dim, fontSize: 12, padding: 40, textAlign: 'center' }}>
              Downloading model header…
            </div>
          )}

          {preview && !previewLoading && (
            <div>
              {/* Model info card */}
              <div style={{
                background: C.surface, border: `1px solid ${C.border}`,
                borderRadius: 10, padding: 14, fontFamily: 'monospace', fontSize: 12, marginBottom: 16,
              }}>
                <div style={{ color: C.accent, fontWeight: 600, fontSize: 13, marginBottom: 8 }}>
                  {selectedFile?.rfilename}
                </div>
                {([
                  ['Architecture', preview.architecture],
                  ['Total blocks', preview.totalBlocks],
                  ['Hidden size', preview.hiddenSize],
                  ['Heads', preview.headCount],
                  ['Full model', fmt(estimate?.fullSize ?? selectedFile?.size ?? 0)],
                ] as [string, string | number][]).map(([label, value]) => (
                  <div key={label} style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', borderBottom: `1px solid ${C.border}` }}>
                    <span style={{ color: C.dim }}>{label}</span>
                    <span style={{ color: C.text }}>{value}</span>
                  </div>
                ))}
              </div>

              {/* Block range picker */}
              <div style={{
                background: C.surface, border: `1px solid ${C.border}`,
                borderRadius: 10, padding: 14, marginBottom: 16,
              }}>
                <div style={{ fontSize: 12, color: C.text, fontWeight: 600, marginBottom: 10 }}>
                  Select blocks to download
                </div>
                <div style={{ fontSize: 11, color: C.dim, marginBottom: 12 }}>
                  Only download the transformer blocks you want to host. Other peers can host the rest.
                </div>

                <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12 }}>
                  <label style={{ color: C.dim, fontSize: 12 }}>Start</label>
                  <input
                    type="number"
                    min={0}
                    max={preview.totalBlocks - 1}
                    value={pBlockStart}
                    onChange={e => {
                      const v = Math.max(0, Math.min(Number(e.target.value), preview.totalBlocks - 1))
                      updateEstimate(v, Math.max(v, pBlockEnd))
                    }}
                    style={numInputStyle}
                  />
                  <label style={{ color: C.dim, fontSize: 12 }}>End</label>
                  <input
                    type="number"
                    min={0}
                    max={preview.totalBlocks - 1}
                    value={pBlockEnd}
                    onChange={e => {
                      const v = Math.max(pBlockStart, Math.min(Number(e.target.value), preview.totalBlocks - 1))
                      updateEstimate(pBlockStart, v)
                    }}
                    style={numInputStyle}
                  />
                  <span style={{ color: C.dim, fontSize: 11 }}>
                    {pBlockEnd - pBlockStart + 1} of {preview.totalBlocks} blocks
                  </span>
                </div>

                {/* Block range visualisation */}
                <div style={{ display: 'flex', gap: 2, marginBottom: 12 }}>
                  {Array.from({ length: preview.totalBlocks }, (_, i) => (
                    <div
                      key={i}
                      style={{
                        flex: 1, height: 14, borderRadius: 2,
                        background: i >= pBlockStart && i <= pBlockEnd ? C.accent : C.border,
                        opacity: i >= pBlockStart && i <= pBlockEnd ? 1 : 0.4,
                        transition: 'background 0.15s, opacity 0.15s',
                      }}
                      title={`Block ${i}`}
                    />
                  ))}
                </div>

                {/* Size estimate */}
                {estimate && (
                  <div style={{
                    background: C.green + '11', border: `1px solid ${C.green}33`,
                    borderRadius: 8, padding: '10px 14px', fontSize: 12,
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <span style={{ color: C.dim }}>Partial download</span>
                      <span style={{ color: C.green, fontWeight: 600 }}>{fmt(estimate.partialSize)}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <span style={{ color: C.dim }}>Full model</span>
                      <span style={{ color: C.dim }}>{fmt(estimate.fullSize)}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: C.dim }}>Savings</span>
                      <span style={{ color: C.green, fontWeight: 600 }}>
                        {Math.round(estimate.savedPercent)}% smaller
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Download buttons */}
              <button
                onClick={() => startPartialDownload(false)}
                style={{
                  display: 'block', width: '100%', marginBottom: 8,
                  background: C.accent, color: '#fff', border: 'none',
                  borderRadius: 8, padding: '12px 20px', fontSize: 13,
                  cursor: 'pointer', fontWeight: 600,
                }}
              >
                Download {pBlockEnd - pBlockStart + 1} blocks
                {estimate ? ` (${fmt(estimate.partialSize)})` : ''}
              </button>

              <button
                onClick={() => startPartialDownload(true)}
                style={{
                  display: 'block', width: '100%',
                  background: 'transparent', color: C.dim,
                  border: `1px solid ${C.border}`, borderRadius: 8,
                  padding: '10px 20px', fontSize: 12, cursor: 'pointer',
                }}
              >
                Download full model instead ({fmt(estimate?.fullSize ?? selectedFile?.size ?? 0)})
              </button>
            </div>
          )}
        </div>
      )}

      {/* ── Downloading ───────────────────────────────────────────── */}
      {view === 'downloading' && (
        <div>
          <div style={{
            background: C.surface, border: `1px solid ${C.border}`,
            borderRadius: 10, padding: 20, textAlign: 'center',
          }}>
            <div style={{ color: C.text, fontSize: 13, fontWeight: 600, marginBottom: 6 }}>
              Downloading from Hugging Face
            </div>
            <div style={{ color: C.dim, fontSize: 11, marginBottom: 16, fontFamily: 'monospace' }}>
              {progress?.file ?? '…'}
            </div>

            {/* Progress bar */}
            <div style={{
              width: '100%', height: 8, background: C.border,
              borderRadius: 4, overflow: 'hidden', marginBottom: 8,
            }}>
              <div style={{
                width: `${progress?.percent ?? 0}%`, height: '100%',
                background: C.accent, borderRadius: 4,
                transition: 'width 0.3s ease',
              }} />
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: C.dim, marginBottom: 16 }}>
              <span>{progress ? fmt(progress.downloadedBytes) : '0 KB'}</span>
              <span>{progress?.percent ?? 0}%</span>
              <span>{progress ? fmt(progress.totalBytes) : '?'}</span>
            </div>

            <button
              onClick={cancelDownload}
              style={{
                background: 'transparent', color: C.red,
                border: `1px solid ${C.red}44`,
                borderRadius: 8, padding: '8px 18px', fontSize: 12, cursor: 'pointer',
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

const backBtnStyle: React.CSSProperties = {
  background: 'transparent', color: C.dim,
  border: `1px solid ${C.border}`, borderRadius: 6,
  padding: '4px 12px', fontSize: 11, cursor: 'pointer',
}

const numInputStyle: React.CSSProperties = {
  width: 60, background: C.surface, color: C.text,
  border: `1px solid ${C.border}`, borderRadius: 6,
  padding: '6px 8px', fontSize: 12, textAlign: 'center',
  outline: 'none',
}
