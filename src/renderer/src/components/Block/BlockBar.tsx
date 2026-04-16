import React from 'react'

interface BlockBarProps {
  totalBlocks: number
  ranges: { start: number; end: number; type: 'local' | 'remote' | 'selected' | 'missing' }[]
  height?: number
}

const TYPE_COLORS: Record<string, string> = {
  local: 'var(--green)',
  remote: 'var(--blue)',
  selected: 'var(--accent)',
  missing: 'var(--border)',
}

export default function BlockBar({ totalBlocks, ranges, height = 12 }: BlockBarProps): React.JSX.Element {
  if (totalBlocks === 0) return <div />
  const blocks = Array<string>(totalBlocks).fill('missing')
  for (const range of ranges) {
    for (let i = Math.max(0, range.start); i <= Math.min(range.end, totalBlocks - 1); i++) {
      blocks[i] = range.type
    }
  }
  return (
    <div style={{ display: 'flex', gap: 1 }}>
      {blocks.map((type, i) => (
        <div key={i} title={`Block ${i}`} style={{
          flex: 1, height, borderRadius: 2, minWidth: 2,
          background: TYPE_COLORS[type] ?? TYPE_COLORS.missing,
          opacity: type === 'missing' ? 0.25 : 1,
          transition: 'background 0.15s, opacity 0.15s',
        }} />
      ))}
    </div>
  )
}
