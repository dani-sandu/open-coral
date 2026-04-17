import React, { useState, useEffect, useCallback } from 'react'
import type { NetworkModelEntry, ModelInfo } from '../../types'
import StatusDot from '../shared/StatusDot'
import cmp from '../shared/components.module.css'
import { useToast } from '../Toast/ToastProvider'

interface ModelSelectorProps {
  currentModel: ModelInfo | null
  onModelLoaded: (model: ModelInfo) => void
}

function displayName(hfFilename: string): string {
  return hfFilename.endsWith('.gguf') ? hfFilename.slice(0, -5) : hfFilename
}

function coverageDotColor(entry: NetworkModelEntry): 'green' | 'yellow' | 'red' {
  if (entry.complete) return 'green'
  if (entry.coveredBlocks > 0) return 'yellow'
  return 'red'
}

export default function ModelSelector({ currentModel, onModelLoaded }: ModelSelectorProps): React.JSX.Element {
  const [models, setModels] = useState<NetworkModelEntry[]>([])
  const [loading, setLoading] = useState(false)
  const [expanded, setExpanded] = useState(true)
  const [loadingId, setLoadingId] = useState<string | null>(null)
  const { addToast } = useToast()

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
      addToast(msg, 'error')
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
    <div style={{ fontSize: 'var(--fs-md)', fontFamily: 'var(--font-ui)' }}>
      {/* Header row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <button
          onClick={() => setExpanded(e => !e)}
          style={{
            background: 'transparent',
            color: 'var(--text)',
            border: 'none',
            padding: 0,
            cursor: 'pointer',
            fontSize: 'var(--fs-md)',
            display: 'flex',
            alignItems: 'center',
            gap: 5,
          }}
        >
          <span style={{ color: 'var(--dim)', fontSize: 'var(--fs-xs)' }}>{expanded ? '▾' : '▸'}</span>
          <span style={{ color: 'var(--accent)', fontWeight: 600 }}>Network Models</span>
          {models.length > 0 && (
            <span style={{ color: 'var(--dim)' }}>({models.length})</span>
          )}
        </button>
        <button
          onClick={discover}
          disabled={loading}
          className={cmp.btnSecondary}
        >
          {loading ? '…' : '↻ Refresh'}
        </button>
      </div>

      {/* Model list */}
      {expanded && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {models.length === 0 && !loading && (
            <div style={{ color: 'var(--dim)', fontSize: 'var(--fs-sm)', padding: '4px 0' }}>
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
                  background: active
                    ? 'color-mix(in srgb, var(--accent) 8%, transparent)'
                    : 'var(--bg)',
                  border: active
                    ? '1px solid color-mix(in srgb, var(--accent) 27%, transparent)'
                    : '1px solid var(--border)',
                  borderRadius: 'var(--radius-md)',
                }}
              >
                {/* Coverage dot */}
                <StatusDot
                  color={dotColor}
                />

                {/* Model info */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div
                    style={{
                      color: active ? 'var(--accent)' : 'var(--text)',
                      fontWeight: active ? 600 : 400,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                    title={entry.hfFilename}
                  >
                    {displayName(entry.hfFilename)}
                  </div>
                  <div style={{ color: 'var(--dim)', fontSize: 'var(--fs-xs)', marginTop: 1 }}>
                    {entry.repoId}
                    {' · '}
                    {coveragePct}% coverage
                    {' · '}
                    {entry.peers.length} peer{entry.peers.length !== 1 ? 's' : ''}
                  </div>
                </div>

                {/* Action button */}
                {active ? (
                  <span style={{ color: 'var(--green)', fontSize: 'var(--fs-sm)', flexShrink: 0 }}>Active</span>
                ) : (
                  <button
                    onClick={() => handleUse(entry)}
                    disabled={isLoading}
                    className={cmp.btnPrimary}
                    style={{ padding: '3px 10px', fontSize: 'var(--fs-sm)', flexShrink: 0 }}
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
