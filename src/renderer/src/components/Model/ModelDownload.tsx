import React, { useEffect } from 'react'
import type { DownloadProgress } from '../../types'
import { fmt } from '../shared/format-utils'
import cmp from '../shared/components.module.css'
import { useAnimeRef, DURATION_PROGRESS } from '../../lib/anime'

export interface ModelDownloadProps {
  progress: DownloadProgress | null
  onCancel: () => void
}

export default function ModelDownload({
  progress, onCancel,
}: ModelDownloadProps): React.JSX.Element {
  const [barRef, animateBar] = useAnimeRef<HTMLDivElement>()
  const pct = progress?.percent ?? 0

  useEffect(() => {
    animateBar({
      width: `${pct}%`,
      ease: 'outBack(1.5)',
      duration: DURATION_PROGRESS,
    })
    if (pct >= 100) {
      animateBar({
        scaleY: [1, 1.35, 1],
        ease: 'outElastic(1, 0.5)',
        duration: 500,
        delay: DURATION_PROGRESS,
      })
    }
  }, [pct])

  return (
    <div>
      <div className={cmp.card} style={{
        borderRadius: 'var(--radius-xl)', padding: 20, textAlign: 'center',
      }}>
        <div style={{ color: 'var(--text)', fontSize: 'var(--fs-lg)', fontWeight: 600, marginBottom: 6 }}>
          Downloading from Hugging Face
        </div>
        <div style={{ color: 'var(--dim)', fontSize: 'var(--fs-sm)', marginBottom: 16, fontFamily: 'var(--font-mono)' }}>
          {progress?.totalShards && progress.totalShards > 1
            ? `Shard ${progress.currentShard ?? 1} of ${progress.totalShards} — ${progress.file}`
            : (progress?.file ?? '')}
        </div>

        <div style={{
          width: '100%', height: 8, background: 'var(--border)',
          borderRadius: 'var(--radius-sm)', overflow: 'hidden', marginBottom: 8,
        }}>
          <div
            ref={barRef}
            style={{
              width: `${pct}%`, height: '100%',
              background: 'var(--accent)', borderRadius: 'var(--radius-sm)',
              transformOrigin: 'left center',
            }}
          />
        </div>

        <div style={{
          display: 'flex', justifyContent: 'space-between',
          fontSize: 'var(--fs-sm)', color: 'var(--dim)', marginBottom: 16,
        }}>
          <span>{progress ? fmt(progress.downloadedBytes) : '0 KB'}</span>
          <span>{pct}%</span>
          <span>{progress ? fmt(progress.totalBytes) : '?'}</span>
        </div>

        <button onClick={onCancel} className={cmp.btnDanger}>
          Cancel
        </button>
      </div>
    </div>
  )
}
