import React, { useState, useCallback } from 'react'

const C = {
  surface: '#181825', border: '#313244',
  text: '#cdd6f4', dim: '#6c7086', accent: '#7c6af7',
  green: '#a6e3a1', red: '#f38ba8', yellow: '#f9e2af', blue: '#89b4fa',
}

function shortId(id: string): string {
  if (id === 'local') return 'local'
  return id.slice(0, 8) + '…'
}

export default function InferencePanel(): React.JSX.Element {
  const [nTokens, setNTokens] = useState(2)
  const [result, setResult] = useState<InferenceDemoResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [running, setRunning] = useState(false)

  const run = useCallback(async () => {
    setRunning(true)
    setError(null)
    setResult(null)
    try {
      const r = await window.coral.runInferenceDemo(nTokens)
      setResult(r)
    } catch (e) {
      setError(String(e))
    } finally {
      setRunning(false)
    }
  }, [nTokens])

  return (
    <div style={{ padding: 20, fontFamily: 'system-ui' }}>
      <h2 style={{ color: C.text, fontSize: 16, margin: '0 0 16px' }}>
        <span style={{ color: C.accent }}>⬡</span> Inference Demo
      </h2>

      <div style={{ fontSize: 12, color: C.dim, marginBottom: 16, lineHeight: 1.5 }}>
        Sends a random Float32 tensor through the full block chain (local + remote peers).
        Demonstrates P2P tensor routing end-to-end.
      </div>

      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 20 }}>
        <label style={{ color: C.dim, fontSize: 12 }}>
          nTokens
          <input
            type="number" min={1} max={16} value={nTokens}
            onChange={e => setNTokens(Number(e.target.value))}
            style={{ marginLeft: 8, width: 50, background: C.surface, color: C.text, border: `1px solid ${C.border}`, borderRadius: 4, padding: '4px 6px', fontSize: 12 }}
          />
        </label>
        <button
          onClick={run} disabled={running}
          style={{
            background: C.accent, color: '#fff', border: 'none',
            borderRadius: 8, padding: '10px 20px', fontSize: 13,
            cursor: running ? 'default' : 'pointer', opacity: running ? 0.6 : 1,
          }}
        >
          {running ? 'Running…' : 'Run Demo'}
        </button>
      </div>

      {error && (
        <div style={{
          background: C.surface, border: `1px solid ${C.red}44`,
          borderRadius: 8, padding: 12, color: C.red, fontSize: 12, marginBottom: 16,
        }}>
          {error}
        </div>
      )}

      {result && (
        <div>
          {/* Summary */}
          <div style={{
            background: C.surface, border: `1px solid ${C.border}`,
            borderRadius: 10, padding: 14, fontFamily: 'monospace', fontSize: 12, marginBottom: 16,
          }}>
            <div style={{ color: C.green, fontWeight: 600, marginBottom: 8 }}>Chain complete</div>
            {([
              ['Total time', result.totalDurationMs + ' ms'],
              ['Tokens', result.nTokens],
              ['Hidden size', result.nEmbd],
              ['Output norm', result.outputNorm.toFixed(4)],
              ['Steps', result.chainSteps.length],
            ] as [string, string | number][]).map(([l, v]) => (
              <div key={l} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}>
                <span style={{ color: C.dim }}>{l}</span>
                <span style={{ color: C.text }}>{v}</span>
              </div>
            ))}
          </div>

          {/* Chain steps */}
          <div style={{ fontSize: 12 }}>
            <div style={{ color: C.dim, marginBottom: 6 }}>Chain steps:</div>
            {result.chainSteps.map((step, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '6px 10px', marginBottom: 4,
                background: C.surface, borderRadius: 6,
                border: `1px solid ${step.peerId === 'local' ? C.accent + '44' : C.border}`,
                fontFamily: 'monospace',
              }}>
                <span style={{ color: C.dim, minWidth: 20, textAlign: 'right' }}>{i + 1}.</span>
                <span style={{
                  color: step.peerId === 'local' ? C.accent : C.blue,
                  minWidth: 120,
                }}>
                  {shortId(step.peerId)}
                </span>
                <span style={{ color: C.dim }}>
                  blk {step.blockStart}–{step.blockEnd}
                </span>
                <span style={{ marginLeft: 'auto', color: C.yellow }}>
                  {step.durationMs} ms
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {!result && !error && !running && (
        <div style={{ color: C.dim, fontSize: 12 }}>
          Start hosting blocks first, then run the demo.
        </div>
      )}
    </div>
  )
}
