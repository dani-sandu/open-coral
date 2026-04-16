import React from 'react'

interface BlockRangeTagProps {
  start: number
  end: number
  color?: string
}

export default function BlockRangeTag({ start, end, color = 'var(--accent)' }: BlockRangeTagProps): React.JSX.Element {
  return (
    <span style={{
      display: 'inline-block', padding: '1px 6px', borderRadius: 4,
      fontSize: 10, fontFamily: 'var(--font-mono)',
      background: `color-mix(in srgb, ${color} 13%, transparent)`,
      color: color,
      border: `1px solid color-mix(in srgb, ${color} 27%, transparent)`,
      marginRight: 3,
    }}>
      blk {start}&ndash;{end}
    </span>
  )
}
