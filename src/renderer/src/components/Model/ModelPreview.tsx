import React from 'react'
import type { HFModelPreview, HFFileInfo, BlockEstimate } from '../../types'
import { fmt } from '../shared/format-utils'
import cmp from '../shared/components.module.css'

// ── Props ──────────────────────────────────────────────────────────────────────

export interface ModelPreviewProps {
  preview: HFModelPreview | null
  selectedFile: HFFileInfo | null
  repoId: string
  loading: boolean
  blockStart: number
  blockEnd: number
  estimate: BlockEstimate | null
  onBlockRangeChange: (start: number, end: number) => void
  onDownload: (full: boolean) => void
  onBack: () => void
}

// ── Component ──────────────────────────────────────────────────────────────────

export default function ModelPreview({
  preview, selectedFile, repoId, loading,
  blockStart, blockEnd, estimate,
  onBlockRangeChange, onDownload, onBack,
}: ModelPreviewProps): React.JSX.Element {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <button onClick={onBack} className={cmp.btnSecondary}>Back</button>
        <span style={{ color: 'var(--accent)', fontSize: 'var(--fs-md)', fontWeight: 600 }}>{repoId}</span>
      </div>

      {loading && (
        <div style={{ color: 'var(--dim)', fontSize: 'var(--fs-md)', padding: 40, textAlign: 'center' }}>
          Downloading model header
        </div>
      )}

      {preview && !loading && (
        <div>
          <div className={cmp.card} style={{
            fontFamily: 'var(--font-mono)', fontSize: 'var(--fs-md)', marginBottom: 16,
          }}>
            <div style={{ color: 'var(--accent)', fontWeight: 600, fontSize: 'var(--fs-lg)', marginBottom: 8 }}>
              {selectedFile?.rfilename}
            </div>
            {([
              ['Architecture', preview.architecture],
              ['Total blocks', preview.totalBlocks],
              ['Hidden size', preview.hiddenSize],
              ['Heads', preview.headCount],
              ['Full model', fmt(estimate?.fullSize ?? selectedFile?.size ?? 0)],
            ] as [string, string | number][]).map(([label, value]) => (
              <div key={label} style={{
                display: 'flex', justifyContent: 'space-between',
                padding: '3px 0', borderBottom: '1px solid var(--border)',
              }}>
                <span style={{ color: 'var(--dim)' }}>{label}</span>
                <span style={{ color: 'var(--text)' }}>{value}</span>
              </div>
            ))}
          </div>

          <div className={cmp.card} style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 'var(--fs-md)', color: 'var(--text)', fontWeight: 600, marginBottom: 10 }}>
              Select blocks to download
            </div>
            <div style={{ fontSize: 'var(--fs-sm)', color: 'var(--dim)', marginBottom: 12 }}>
              Only download the transformer blocks you want to host. Other peers can host the rest.
            </div>

            <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12 }}>
              <label style={{ color: 'var(--dim)', fontSize: 'var(--fs-md)' }}>Start</label>
              <input
                type="number"
                className={cmp.inputNumber}
                min={0}
                max={preview.totalBlocks - 1}
                value={blockStart}
                onChange={e => {
                  const v = Math.max(0, Math.min(Number(e.target.value), preview.totalBlocks - 1))
                  onBlockRangeChange(v, Math.max(v, blockEnd))
                }}
              />
              <label style={{ color: 'var(--dim)', fontSize: 'var(--fs-md)' }}>End</label>
              <input
                type="number"
                className={cmp.inputNumber}
                min={0}
                max={preview.totalBlocks - 1}
                value={blockEnd}
                onChange={e => {
                  const v = Math.max(blockStart, Math.min(Number(e.target.value), preview.totalBlocks - 1))
                  onBlockRangeChange(blockStart, v)
                }}
              />
              <span style={{ color: 'var(--dim)', fontSize: 'var(--fs-sm)' }}>
                {blockEnd - blockStart + 1} of {preview.totalBlocks} blocks
              </span>
            </div>

            <div style={{ display: 'flex', gap: 2, marginBottom: 12 }}>
              {Array.from({ length: preview.totalBlocks }, (_, i) => (
                <div
                  key={i}
                  style={{
                    flex: 1, height: 14, borderRadius: 2,
                    background: i >= blockStart && i <= blockEnd ? 'var(--accent)' : 'var(--border)',
                    opacity: i >= blockStart && i <= blockEnd ? 1 : 0.4,
                    transition: 'background 0.15s, opacity 0.15s',
                  }}
                  title={`Block ${i}`}
                />
              ))}
            </div>

            {estimate && (
              <div style={{
                background: 'color-mix(in srgb, var(--green) 7%, transparent)',
                border: '1px solid color-mix(in srgb, var(--green) 20%, transparent)',
                borderRadius: 'var(--radius-lg)', padding: '10px 14px', fontSize: 'var(--fs-md)',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ color: 'var(--dim)' }}>Partial download</span>
                  <span style={{ color: 'var(--green)', fontWeight: 600 }}>{fmt(estimate.partialSize)}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ color: 'var(--dim)' }}>Full model</span>
                  <span style={{ color: 'var(--dim)' }}>{fmt(estimate.fullSize)}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--dim)' }}>Savings</span>
                  <span style={{ color: 'var(--green)', fontWeight: 600 }}>
                    {Math.round(estimate.savedPercent)}% smaller
                  </span>
                </div>
              </div>
            )}
          </div>

          <button
            onClick={() => onDownload(false)}
            className={cmp.btnPrimary}
            style={{
              display: 'block', width: '100%', marginBottom: 8,
              borderRadius: 'var(--radius-lg)',
            }}
          >
            Download {blockEnd - blockStart + 1} blocks
            {estimate ? ` (${fmt(estimate.partialSize)})` : ''}
          </button>

          <button
            onClick={() => onDownload(true)}
            className={cmp.btnSecondary}
            style={{
              display: 'block', width: '100%',
              borderRadius: 'var(--radius-lg)', padding: '10px 20px',
              fontSize: 'var(--fs-md)',
            }}
          >
            Download full model instead ({fmt(estimate?.fullSize ?? selectedFile?.size ?? 0)})
          </button>
        </div>
      )}
    </div>
  )
}
