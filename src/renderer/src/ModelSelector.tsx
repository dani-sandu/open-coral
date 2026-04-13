import React, { useState, useEffect, useCallback } from 'react'
import type { NetworkModelEntry, ModelInfo } from './types'

const C = {
  bg: '#1e1e2e',
  surface: '#181825',
  border: '#313244',
  text: '#cdd6f4',
  dim: '#6c7086',
  accent: '#7c6af7',
  green: '#a6e3a1',
  yellow: '#f9e2af',
  red: '#f38ba8',
}

interface ModelSelectorProps {
  currentModel: ModelInfo | null
  onModelLoaded: (model: ModelInfo) => void
}

function displayName(hfFilename: string): string {
  return hfFilename.endsWith('.gguf') ? hfFilename.slice(0, -5) : hfFilename
}

function coverageDotColor(entry: NetworkModelEntry): string {
  if (entry.complete) return C.green
  if (entry.coveredBlocks > 0) return C.yellow
  return C.red
}

export default function ModelSelector({ currentModel, onModelLoaded }: ModelSelectorProps): React.JSX.Element {
  const [models, setModels] = useState<NetworkModelEntry[]>([])
  const [loading, setLoading] = useState(false)
  const [expanded, setExpanded] = useState(true)
  const [loadingId, setLoadingId] = useState<string | null>(null)

  const discover = useCallback(async () => {
    setLoading(true)
    try {
      const results = await window.opencoral.discoverNetworkModels()
      setModels(results)
    } catch (e) {
      console.error('discoverNetworkModels failed:', e)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    discover()
  }, [discover])

  const handleUse = useCallback(async (entry: NetworkModelEntry) => {
    const id = entry.repoId + '/' + entry.hfFilename
    setLoadingId(id)
    try {
      const model = await window.opencoral.loadModelByHFIdentity(entry.repoId, entry.hfFilename)
      onModelLoaded(model)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      if (msg.toLowerCase().includes('not found')) {
        alert(`Model not found locally. Download it from the Model tab first.`)
      } else {
        console.error('loadModelByHFIdentity failed:', e)
      }
    } finally {
      setLoadingId(null)
    }
  }, [onModelLoaded])

  const isActive = (entry: NetworkModelEntry): boolean => {
    if (!currentModel) return false
    return !!(
      currentModel.repoId &&
      currentModel.hfFilename &&
      currentModel.repoId === entry.repoId &&
      currentModel.hfFilename === entry.hfFilename
    )
  }

  const entryId = (entry: NetworkModelEntry): string => entry.repoId + '/' + entry.hfFilename

  return (
    <div style={{ fontSize: 12, fontFamily: 'system-ui' }}>
      {/* Header row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <button
          onClick={() => setExpanded(e => !e)}
          style={{
            background: 'transparent',
            color: C.text,
            border: 'none',
            padding: 0,
            cursor: 'pointer',
            fontSize: 12,
            display: 'flex',
            alignItems: 'center',
            gap: 5,
          }}
        >
          <span style={{ color: C.dim, fontSize: 10 }}>{expanded ? '▾' : '▸'}</span>
          <span style={{ color: C.accent, fontWeight: 600 }}>Network Models</span>
          {models.length > 0 && (
            <span style={{ color: C.dim }}>({models.length})</span>
          )}
        </button>
        <button
          onClick={discover}
          disabled={loading}
          style={{
            background: 'transparent',
            color: C.dim,
            border: `1px solid ${C.border}`,
            borderRadius: 4,
            padding: '2px 8px',
            fontSize: 10,
            cursor: loading ? 'default' : 'pointer',
            opacity: loading ? 0.6 : 1,
          }}
        >
          {loading ? '…' : '↻ Refresh'}
        </button>
      </div>

      {/* Model list */}
      {expanded && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {models.length === 0 && !loading && (
            <div style={{ color: C.dim, fontSize: 11, padding: '4px 0' }}>
              No models found on the network. Load a model and start hosting in the Models tab, or connect to peers that are hosting.
            </div>
          )}
          {models.map(entry => {
            const active = isActive(entry)
            const id = entryId(entry)
            const isLoading = loadingId === id
            const dotColor = coverageDotColor(entry)
            const coveragePct = entry.totalBlocks > 0
              ? Math.round((entry.coveredBlocks / entry.totalBlocks) * 100)
              : 0

            return (
              <div
                key={id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '5px 8px',
                  background: active ? C.accent + '15' : C.bg,
                  border: `1px solid ${active ? C.accent + '44' : C.border}`,
                  borderRadius: 6,
                }}
              >
                {/* Coverage dot */}
                <div
                  title={entry.complete ? 'Full coverage' : entry.coveredBlocks > 0 ? 'Partial coverage' : 'No coverage'}
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: dotColor,
                    flexShrink: 0,
                  }}
                />

                {/* Model info */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div
                    style={{
                      color: active ? C.accent : C.text,
                      fontWeight: active ? 600 : 400,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                    title={entry.hfFilename}
                  >
                    {displayName(entry.hfFilename)}
                  </div>
                  <div style={{ color: C.dim, fontSize: 10, marginTop: 1 }}>
                    {entry.repoId}
                    {' · '}
                    {coveragePct}% coverage
                    {' · '}
                    {entry.peers.length} peer{entry.peers.length !== 1 ? 's' : ''}
                  </div>
                </div>

                {/* Action button */}
                {active ? (
                  <span style={{ color: C.green, fontSize: 11, flexShrink: 0 }}>Active</span>
                ) : (
                  <button
                    onClick={() => handleUse(entry)}
                    disabled={isLoading}
                    style={{
                      background: C.accent,
                      color: '#fff',
                      border: 'none',
                      borderRadius: 4,
                      padding: '3px 10px',
                      fontSize: 11,
                      cursor: isLoading ? 'default' : 'pointer',
                      opacity: isLoading ? 0.6 : 1,
                      flexShrink: 0,
                    }}
                  >
                    {isLoading ? '…' : 'Use'}
                  </button>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
