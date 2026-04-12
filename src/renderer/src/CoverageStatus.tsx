import React from 'react'
import type { CoverageReport } from './types'

const C = {
  surface: '#181825',
  border: '#313244',
  text: '#cdd6f4',
  dim: '#6c7086',
  accent: '#7c6af7',
  green: '#a6e3a1',
  red: '#f38ba8',
  blue: '#89b4fa',
}

interface Props {
  report: CoverageReport | null
  loading?: boolean
  onRefresh?: () => void
}

export default function CoverageStatus({ report, loading, onRefresh }: Props): React.JSX.Element {
  if (loading) {
    return (
      <div style={{ fontSize: 11, color: C.dim, fontFamily: 'monospace' }}>
        Checking coverage…
      </div>
    )
  }

  if (!report || report.totalBlocks === 0) {
    return (
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 11, color: C.dim }}>Load a model to check coverage.</span>
        {onRefresh && (
          <button onClick={onRefresh} style={refreshBtnStyle}>↻</button>
        )}
      </div>
    )
  }

  // Build per-block color array: green = local, blue = remote, red = missing
  const blockColors = Array<string>(report.totalBlocks).fill(C.red + '55')
  for (const cov of report.covered) {
    const color = cov.peerId === 'local' ? C.green : C.blue
    for (let i = Math.max(0, cov.blockStart); i <= Math.min(cov.blockEnd, report.totalBlocks - 1); i++) {
      blockColors[i] = color
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {/* Status text + refresh button */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 11, fontFamily: 'monospace' }}>
          {report.complete
            ? <span style={{ color: C.green }}>● All {report.totalBlocks} blocks covered</span>
            : <span style={{ color: C.dim }}>
                {report.totalBlocks - report.missing.length}/{report.totalBlocks} blocks available
              </span>
          }
        </span>
        {onRefresh && (
          <button onClick={onRefresh} style={refreshBtnStyle}>↻</button>
        )}
      </div>

      {/* Block bar */}
      <div style={{ display: 'flex', gap: 1, height: 8 }}>
        {blockColors.map((color, i) => (
          <div
            key={i}
            title={`Block ${i}`}
            style={{
              flex: 1,
              borderRadius: 1,
              background: color,
              minWidth: 2,
            }}
          />
        ))}
      </div>

      {/* Legend */}
      <div style={{ display: 'flex', gap: 12, fontSize: 10, color: C.dim }}>
        <span><span style={{ color: C.green }}>■</span> local</span>
        <span><span style={{ color: C.blue }}>■</span> remote</span>
        <span><span style={{ color: C.red }}>■</span> missing</span>
      </div>
    </div>
  )
}

const refreshBtnStyle: React.CSSProperties = {
  background: 'transparent',
  color: '#6c7086',
  border: '1px solid #313244',
  borderRadius: 4,
  padding: '2px 8px',
  fontSize: 10,
  cursor: 'pointer',
}
