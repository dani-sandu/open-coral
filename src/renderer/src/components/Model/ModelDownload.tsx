import React from 'react'
import type { DownloadProgress } from '../../types'
import { fmt } from '../shared/format-utils'
import cmp from '../shared/components.module.css'

// ── Props ──────────────────────────────────────────────────────────────────────

export interface ModelDownloadProps {
  progress: DownloadProgress | null
  onCancel: () => void
}

// ── Component ──────────────────────────────────────────────────────────────────

export default function ModelDownload({
  progress, onCancel,
}: ModelDownloadProps): React.JSX.Element {
  return (
    <div>
      <div className={cmp.card} style={{
        borderRadius: 'var(--radius-xl)', padding: 20, textAlign: 'center',
      }}>
        <div style={{ color: 'var(--text)', fontSize: 'var(--fs-lg)', fontWeight: 600, marginBottom: 6 }}>
          Downloading from Hugging Face
        </div>
        <div style={{ color: 'var(--dim)', fontSize: 'var(--fs-sm)', marginBottom: 16, fontFamily: 'var(--font-mono)' }}>
          {progress?.file ?? ''}
        </div>

        <div style={{
          width: '100%', height: 8, background: 'var(--border)',
          borderRadius: 'var(--radius-sm)', overflow: 'hidden', marginBottom: 8,
        }}>
          <div style={{
            width: `${progress?.percent ?? 0}%`, height: '100%',
            background: 'var(--accent)', borderRadius: 'var(--radius-sm)',
            transition: 'width 0.3s ease',
          }} />
        </div>

        <div style={{
          display: 'flex', justifyContent: 'space-between',
          fontSize: 'var(--fs-sm)', color: 'var(--dim)', marginBottom: 16,
        }}>
          <span>{progress ? fmt(progress.downloadedBytes) : '0 KB'}</span>
          <span>{progress?.percent ?? 0}%</span>
          <span>{progress ? fmt(progress.totalBytes) : '?'}</span>
        </div>

        <button onClick={onCancel} className={cmp.btnDanger}>
          Cancel
        </button>
      </div>
    </div>
  )
}
