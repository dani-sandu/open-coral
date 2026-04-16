import React from 'react'
import type { HFModelResult } from '../../types'
import { fmtCount } from '../shared/format-utils'
import cmp from '../shared/components.module.css'

// ── Props ──────────────────────────────────────────────────────────────────────

export interface ModelSearchProps {
  results: HFModelResult[]
  query: string
  onSelectRepo: (repo: HFModelResult) => void
  onBack: () => void
}

// ── Component ──────────────────────────────────────────────────────────────────

export default function ModelSearch({
  results, query, onSelectRepo, onBack,
}: ModelSearchProps): React.JSX.Element {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <button onClick={onBack} className={cmp.btnSecondary}>Back</button>
        <span style={{ color: 'var(--dim)', fontSize: 'var(--fs-sm)' }}>{results.length} results</span>
      </div>

      {results.length === 0 && (
        <div style={{ color: 'var(--dim)', fontSize: 'var(--fs-md)', padding: 20, textAlign: 'center' }}>
          No GGUF models found for &ldquo;{query}&rdquo;
        </div>
      )}

      {results.map(repo => (
        <button
          key={repo.id}
          onClick={() => onSelectRepo(repo)}
          className={cmp.card}
          style={{
            display: 'block', width: '100%', textAlign: 'left',
            borderRadius: 'var(--radius-lg)', marginBottom: 6, cursor: 'pointer',
            transition: 'border-color 0.15s',
          }}
          onMouseEnter={e => (e.currentTarget.style.borderColor = 'color-mix(in srgb, var(--accent) 40%, transparent)')}
          onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--border)')}
        >
          <div style={{ color: 'var(--text)', fontSize: 'var(--fs-lg)', fontWeight: 600, marginBottom: 4 }}>
            {repo.id}
          </div>
          <div style={{ display: 'flex', gap: 14, fontSize: 'var(--fs-sm)', color: 'var(--dim)' }}>
            <span>Downloads: {fmtCount(repo.downloads)}</span>
            <span>Likes: {fmtCount(repo.likes)}</span>
          </div>
        </button>
      ))}
    </div>
  )
}
