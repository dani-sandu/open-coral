import React from 'react'

const COLOR_MAP: Record<string, string> = {
  green: 'var(--green)',
  yellow: 'var(--yellow)',
  red: 'var(--red)',
  blue: 'var(--blue)',
}

interface StatusDotProps {
  color: 'green' | 'yellow' | 'red' | 'blue'
  label?: string
}

export default function StatusDot({ color, label }: StatusDotProps): React.JSX.Element {
  const cssColor = COLOR_MAP[color]
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
      <span style={{
        width: 8, height: 8, borderRadius: '50%',
        background: cssColor, display: 'inline-block', flexShrink: 0,
      }} />
      {label && <span>{label}</span>}
    </span>
  )
}
