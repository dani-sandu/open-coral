import React, { useState, useCallback, useEffect } from 'react'
import type { ModelInfo, HostingState } from './types'

const C = {
  surface: '#181825', border: '#313244',
  text: '#cdd6f4', dim: '#6c7086', accent: '#7c6af7',
  green: '#a6e3a1', red: '#f38ba8',
}

export default function BlockHostPanel(): React.JSX.Element {
  const [model, setModel] = useState<ModelInfo | null>(null)
  const [state, setState] = useState<HostingState | null>(null)
  const [blockStart, setStart] = useState(0)
  const [blockEnd, setEnd] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const reload = useCallback(async () => {
    const [m, s] = await Promise.all([
      window.opencoral.getModel(),
      window.opencoral.getHostingState(),
    ])
    setModel(m)
    setState(s)
    if (m && !s) {
      setStart(0)
      setEnd(Math.max(0, Math.floor(m.totalBlocks / 2) - 1))
    }
  }, [])

  useEffect(() => { reload() }, [reload])

  const start = useCallback(async () => {
    setBusy(true)
    setError(null)
    try {
      await window.opencoral.startHosting(blockStart, blockEnd)
      await reload()
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }, [blockStart, blockEnd, reload])

  const stop = useCallback(async () => {
    setBusy(true)
    setError(null)
    try {
      await window.opencoral.stopHosting()
      await reload()
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }, [reload])

  const maxBlock = (model?.totalBlocks ?? 2) - 1

  return (
    <div style={{ padding: 20, fontFamily: 'system-ui' }}>
      <h2 style={{ color: C.text, fontSize: 16, margin: '0 0 16px' }}>
        <span style={{ color: C.accent }}>⬡</span> Block Hosting
      </h2>

      {!model && (
        <div style={{ color: C.dim, fontSize: 12 }}>Load a model first (Model tab).</div>
      )}

      {model && !state && (
        <div>
          <div style={{ marginBottom: 16, fontSize: 12, color: C.dim }}>
            Model has <strong style={{ color: C.text }}>{model.totalBlocks}</strong> blocks.
            Choose which to host locally:
          </div>

          <div style={{ display: 'flex', gap: 16, marginBottom: 16, alignItems: 'center' }}>
            <label style={{ color: C.dim, fontSize: 12, minWidth: 80 }}>
              Block start
              <input
                type="number" min={0} max={blockEnd} value={blockStart}
                onChange={e => setStart(Number(e.target.value))}
                style={{ marginLeft: 8, width: 60, background: C.surface, color: C.text, border: `1px solid ${C.border}`, borderRadius: 4, padding: '4px 6px' }}
              />
            </label>
            <label style={{ color: C.dim, fontSize: 12, minWidth: 80 }}>
              Block end
              <input
                type="number" min={blockStart} max={maxBlock} value={blockEnd}
                onChange={e => setEnd(Number(e.target.value))}
                style={{ marginLeft: 8, width: 60, background: C.surface, color: C.text, border: `1px solid ${C.border}`, borderRadius: 4, padding: '4px 6px' }}
              />
            </label>
          </div>

          <button
            onClick={start} disabled={busy}
            style={{
              background: C.accent, color: '#fff', border: 'none',
              borderRadius: 8, padding: '10px 20px', fontSize: 13,
              cursor: busy ? 'default' : 'pointer', opacity: busy ? 0.6 : 1,
            }}
          >
            {busy ? 'Starting…' : 'Start Hosting'}
          </button>
        </div>
      )}

      {state && (
        <div>
          <div style={{
            background: C.surface, border: `1px solid ${C.border}`,
            borderRadius: 10, padding: 14, fontFamily: 'monospace', fontSize: 12, marginBottom: 16,
          }}>
            <div style={{ color: C.green, fontWeight: 600, marginBottom: 8 }}>
              Hosting active
            </div>
            {([
              ['Blocks', `${state.blockStart} – ${state.blockEnd}`],
              ['Model', state.modelPath.split(/[\\/]/).pop()!],
              ['Hidden', state.hiddenSize],
            ] as [string, string | number][]).map(([l, v]) => (
              <div key={l} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0', color: C.dim }}>
                <span>{l}</span>
                <span style={{ color: C.text }}>{v}</span>
              </div>
            ))}
          </div>

          <button
            onClick={stop} disabled={busy}
            style={{
              background: 'transparent', color: C.red,
              border: `1px solid ${C.red}44`,
              borderRadius: 8, padding: '8px 18px', fontSize: 12,
              cursor: busy ? 'default' : 'pointer',
            }}
          >
            {busy ? 'Stopping…' : 'Stop Hosting'}
          </button>
        </div>
      )}

      {error && (
        <div style={{ color: C.red, fontSize: 11, marginTop: 10 }}>{error}</div>
      )}
    </div>
  )
}
