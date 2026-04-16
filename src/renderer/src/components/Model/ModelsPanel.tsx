import React, { useState, useCallback, useEffect, useRef } from 'react'
import type {
  ModelInfo, LocalModelEntry, HostingState,
  HFModelResult, HFFileInfo, HFModelPreview, DownloadProgress, BlockEstimate,
} from '../../types'
import { extractQuant, getQuantInfo } from './ModelFiles'
import TabShell from '../shared/TabShell'
import ModelHome from './ModelHome'
import ModelSearch from './ModelSearch'
import ModelFiles from './ModelFiles'
import ModelPreview from './ModelPreview'
import ModelDownload from './ModelDownload'
import cmp from '../shared/components.module.css'

// ── Sub-view type ──────────────────────────────────────────────────────────────

type View = 'home' | 'search' | 'files' | 'preview' | 'downloading'

// ── Component ──────────────────────────────────────────────────────────────────

export default function ModelsPanel(): React.JSX.Element {
  // ── Local model list ──────────────────────────────────────────────────────
  const [localModels, setLocalModels] = useState<LocalModelEntry[]>([])
  const [selectedModel, setSelectedModel] = useState<LocalModelEntry | null>(null)
  const [activeModel, setActiveModel] = useState<ModelInfo | null>(null)

  // ── Hosting ───────────────────────────────────────────────────────────────
  const [hostingState, setHostingState] = useState<HostingState | null>(null)
  const [hostBlockStart, setHostBlockStart] = useState(0)
  const [hostBlockEnd, setHostBlockEnd] = useState(0)
  const [hostBusy, setHostBusy] = useState(false)

  // ── General ───────────────────────────────────────────────────────────────
  const [error, setError] = useState<string | null>(null)
  const [loadingModels, setLoadingModels] = useState(true)

  // ── HF search / download flow ─────────────────────────────────────────────
  const [view, setView] = useState<View>('home')
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<HFModelResult[]>([])
  const [searching, setSearching] = useState(false)
  const [selectedRepo, setSelectedRepo] = useState<HFModelResult | null>(null)
  const [files, setFiles] = useState<HFFileInfo[]>([])
  const [loadingFiles, setLoadingFiles] = useState(false)
  const [showAllFiles, setShowAllFiles] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [progress, setProgress] = useState<DownloadProgress | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const [preview, setPreview] = useState<HFModelPreview | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [selectedFile, setSelectedFile] = useState<HFFileInfo | null>(null)
  const [pBlockStart, setPBlockStart] = useState(0)
  const [pBlockEnd, setPBlockEnd] = useState(0)
  const [estimate, setEstimate] = useState<BlockEstimate | null>(null)
  const [loading, setLoading] = useState(false)

  // ── Refresh all state ─────────────────────────────────────────────────────
  const refresh = useCallback(async () => {
    setLoadingModels(true)
    try {
      const [models, model, hosting] = await Promise.all([
        window.opencoral.listLocalModels(),
        window.opencoral.getModel(),
        window.opencoral.getHostingState(),
      ])
      setLocalModels(models)
      setActiveModel(model)
      setHostingState(hosting)
    } finally {
      setLoadingModels(false)
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  // ── Poll download progress ────────────────────────────────────────────────
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
            setActiveModel(m)
            await refresh()
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
  }, [downloading, refresh])

  // ── Model selection / loading ─────────────────────────────────────────────
  const selectLocalModel = useCallback(async (entry: LocalModelEntry | null) => {
    if (!entry) { setSelectedModel(null); return }
    setSelectedModel(entry)
    setError(null)
    const bStart = entry.blockStart ?? 0
    const bEnd = entry.blockEnd ?? Math.max(0, Math.floor(entry.totalBlocks / 2) - 1)
    setHostBlockStart(bStart)
    setHostBlockEnd(bEnd)
    try {
      const m = await window.opencoral.loadModelPath(entry.path)
      setActiveModel(m)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  const pick = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const m = await window.opencoral.selectModel()
      if (m) {
        setActiveModel(m)
        await refresh()
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [refresh])

  // ── Hosting controls ──────────────────────────────────────────────────────
  const startHosting = useCallback(async () => {
    setHostBusy(true)
    setError(null)
    try {
      await window.opencoral.startHosting(hostBlockStart, hostBlockEnd)
      await refresh()
    } catch (e) {
      setError(String(e))
    } finally {
      setHostBusy(false)
    }
  }, [hostBlockStart, hostBlockEnd, refresh])

  const stopHosting = useCallback(async () => {
    setHostBusy(true)
    setError(null)
    try {
      await window.opencoral.stopHosting()
      await refresh()
    } catch (e) {
      setError(String(e))
    } finally {
      setHostBusy(false)
    }
  }, [refresh])

  // ── HF actions ────────────────────────────────────────────────────────────
  const search = useCallback(async () => {
    if (!query.trim()) return
    setSearching(true)
    setError(null)
    try {
      const r = await window.opencoral.hfSearch(query.trim())
      setResults(r)
      setView('search')
    } catch (e) { setError(String(e)) }
    finally { setSearching(false) }
  }, [query])

  const selectRepo = useCallback(async (repo: HFModelResult) => {
    setSelectedRepo(repo)
    setLoadingFiles(true)
    setError(null)
    setShowAllFiles(false)
    try {
      const f = await window.opencoral.hfListFiles(repo.id)
      f.sort((a, b) => {
        const qa = extractQuant(a.rfilename)
        const qb = extractQuant(b.rfilename)
        return (qa ? getQuantInfo(qa).order : 50) - (qb ? getQuantInfo(qb).order : 50)
      })
      setFiles(f)
      setView('files')
    } catch (e) { setError(String(e)) }
    finally { setLoadingFiles(false) }
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
      const est = await window.opencoral.hfEstimateBlocks(0, Math.min(3, p.totalBlocks - 1))
      setEstimate(est)
    } catch (e) {
      setError(String(e))
      setView('files')
    } finally { setPreviewLoading(false) }
  }, [selectedRepo])

  const updateEstimate = useCallback(async (start: number, end: number) => {
    setPBlockStart(start)
    setPBlockEnd(end)
    try {
      const est = await window.opencoral.hfEstimateBlocks(start, end)
      setEstimate(est)
    } catch { setEstimate(null) }
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

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <TabShell title="Models">
      <div style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>
      {error && <div className={cmp.error}>{error}</div>}

      {view === 'home' && (
        <ModelHome
          localModels={localModels}
          selectedModel={selectedModel}
          activeModel={activeModel}
          hostingState={hostingState}
          hostBlockStart={hostBlockStart}
          hostBlockEnd={hostBlockEnd}
          loadingModels={loadingModels}
          hostBusy={hostBusy}
          error={error}
          query={query}
          searching={searching}
          loading={loading}
          onSelectModel={selectLocalModel}
          onSetHostBlockStart={setHostBlockStart}
          onSetHostBlockEnd={setHostBlockEnd}
          onStartHosting={startHosting}
          onStopHosting={stopHosting}
          onSetQuery={setQuery}
          onSearch={search}
          onPickLocal={pick}
        />
      )}

      {view === 'search' && (
        <ModelSearch
          results={results}
          query={query}
          onSelectRepo={selectRepo}
          onBack={() => setView('home')}
        />
      )}

      {view === 'files' && selectedRepo && (
        <ModelFiles
          files={files}
          repoId={selectedRepo.id}
          loading={loadingFiles}
          showAll={showAllFiles}
          onToggleShowAll={() => setShowAllFiles(v => !v)}
          onSelectFile={selectFileForPreview}
          onBack={() => setView('search')}
        />
      )}

      {view === 'preview' && (
        <ModelPreview
          preview={preview}
          selectedFile={selectedFile}
          repoId={selectedRepo?.id ?? ''}
          loading={previewLoading}
          blockStart={pBlockStart}
          blockEnd={pBlockEnd}
          estimate={estimate}
          onBlockRangeChange={updateEstimate}
          onDownload={startPartialDownload}
          onBack={() => setView('files')}
        />
      )}

      {view === 'downloading' && (
        <ModelDownload
          progress={progress}
          onCancel={cancelDownload}
        />
      )}
      </div>
    </TabShell>
  )
}
