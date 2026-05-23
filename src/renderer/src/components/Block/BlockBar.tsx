import React, { useRef, useEffect } from 'react'
import { createLayout, stagger } from 'animejs'
import { ANIMATION_ENABLED, SPRING_STATUS, DURATION_STATUS } from '../../lib/anime'

interface BlockBarProps {
  totalBlocks: number
  ranges: { start: number; end: number; type: 'local' | 'remote' | 'selected' | 'missing' }[]
  height?: number
}

const TYPE_COLORS: Record<string, string> = {
  local: 'var(--green)',
  remote: 'var(--blue)',
  selected: 'var(--accent)',
  missing: 'var(--red)',
}

export default function BlockBar({ totalBlocks, ranges, height = 12 }: BlockBarProps): React.JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!ANIMATION_ENABLED || !containerRef.current) return

    const firstActive = ranges.find(r => r.type !== 'missing')
    const centerIdx = firstActive
      ? Math.floor((firstActive.start + firstActive.end) / 2)
      : Math.floor(totalBlocks / 2)

    const layout = createLayout(containerRef.current, {
      enterFrom: {
        scaleY: 0,
        opacity: 0,
        duration: DURATION_STATUS,
        ease: SPRING_STATUS,
        delay: stagger(6, { from: centerIdx }),
      },
    })

    return () => layout.revert()
  }, []) // mount only

  if (totalBlocks === 0) return <div />

  const blocks = Array<string>(totalBlocks).fill('missing')
  for (const range of ranges) {
    for (let i = Math.max(0, range.start); i <= Math.min(range.end, totalBlocks - 1); i++) {
      blocks[i] = range.type
    }
  }

  return (
    <div ref={containerRef} style={{ display: 'flex', gap: 1 }}>
      {blocks.map((type, i) => (
        <div key={i} title={`Block ${i}`} style={{
          flex: 1, height, borderRadius: 2, minWidth: 2,
          background: TYPE_COLORS[type] ?? TYPE_COLORS.missing,
          opacity: type === 'missing' ? 0.6 : 1,
          transformOrigin: 'bottom',
          transition: 'background 0.15s, opacity 0.15s',
        }} />
      ))}
    </div>
  )
}
