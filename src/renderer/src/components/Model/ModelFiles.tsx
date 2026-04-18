import React from 'react'
import type { HFFileInfo, ShardSet } from '../../types'
import { isShardSet } from '../../../../inference/shard-utils'
import { fmt } from '../shared/format-utils'
import cmp from '../shared/components.module.css'

// ── Quantization helpers ───────────────────────────────────────────────────────

interface QuantInfo {
  quant: string
  quality: 'low' | 'medium' | 'high'
  label: string
  order: number
}

export const QUANT_TABLE: Record<string, QuantInfo> = {
  'Q2_K':   { quant: 'Q2_K',   quality: 'low',    label: 'Smallest — fast, lower quality', order: 1 },
  'Q3_K_S': { quant: 'Q3_K_S', quality: 'low',    label: 'Small — fast, lower quality', order: 2 },
  'Q3_K_M': { quant: 'Q3_K_M', quality: 'low',    label: 'Small — balanced for low RAM', order: 3 },
  'Q3_K_L': { quant: 'Q3_K_L', quality: 'medium', label: 'Small-medium', order: 4 },
  'IQ4_XS': { quant: 'IQ4_XS', quality: 'medium', label: 'Compact medium quality', order: 5 },
  'Q4_0':   { quant: 'Q4_0',   quality: 'medium', label: 'Medium — legacy quantization', order: 6 },
  'Q4_K_S': { quant: 'Q4_K_S', quality: 'medium', label: 'Medium — good speed', order: 7 },
  'Q4_K_M': { quant: 'Q4_K_M', quality: 'medium', label: 'Recommended — best balance of size & quality', order: 8 },
  'Q5_0':   { quant: 'Q5_0',   quality: 'high',   label: 'High — legacy quantization', order: 9 },
  'Q5_K_S': { quant: 'Q5_K_S', quality: 'high',   label: 'High quality', order: 10 },
  'Q5_K_M': { quant: 'Q5_K_M', quality: 'high',   label: 'High quality — slightly larger', order: 11 },
  'Q6_K':   { quant: 'Q6_K',   quality: 'high',   label: 'Very high quality — large', order: 12 },
  'Q8_0':   { quant: 'Q8_0',   quality: 'high',   label: 'Near-lossless — very large', order: 13 },
  'F16':    { quant: 'F16',    quality: 'high',   label: 'Full precision fp16 — largest', order: 14 },
  'F32':    { quant: 'F32',    quality: 'high',   label: 'Full precision fp32 — largest', order: 15 },
  'BF16':   { quant: 'BF16',   quality: 'high',   label: 'Full precision bf16 — largest', order: 16 },
}

export const RECOMMENDED_QUANT = 'Q4_K_M'

export function extractQuant(filename: string): string | null {
  const m = filename.match(/[.-]((?:IQ|Q|F|BF)\d+(?:_K)?(?:_[A-Z]+)?|F16|F32|BF16)(?:[.-]|$)/i)
  return m ? m[1].toUpperCase() : null
}

export function getQuantInfo(quant: string): QuantInfo {
  return QUANT_TABLE[quant] ?? { quant, quality: 'medium' as const, label: quant, order: 50 }
}

const qualityColor: Record<string, string> = {
  low: 'var(--yellow)', medium: 'var(--green)', high: 'var(--blue)',
}

// ── Props ──────────────────────────────────────────────────────────────────────

export interface ModelFilesProps {
  files: (HFFileInfo | ShardSet)[]
  repoId: string
  loading: boolean
  showAll: boolean
  onToggleShowAll: () => void
  onSelectFile: (file: HFFileInfo | ShardSet) => void
  onBack: () => void
}

// ── Component ──────────────────────────────────────────────────────────────────

export default function ModelFiles({
  files, repoId, loading, showAll,
  onToggleShowAll, onSelectFile, onBack,
}: ModelFilesProps): React.JSX.Element {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <button onClick={onBack} className={cmp.btnSecondary}>Back</button>
        <span style={{ color: 'var(--accent)', fontSize: 'var(--fs-md)', fontWeight: 600 }}>{repoId}</span>
      </div>

      {loading && (
        <div style={{ color: 'var(--dim)', fontSize: 'var(--fs-md)', padding: 20, textAlign: 'center' }}>Loading files</div>
      )}

      {!loading && files.length === 0 && (
        <div style={{ color: 'var(--dim)', fontSize: 'var(--fs-md)', padding: 20, textAlign: 'center' }}>
          No .gguf files found in this repository.
        </div>
      )}

      {!loading && files.length > 0 && (() => {
        const recommended = files.find(f => extractQuant(isShardSet(f) ? f.canonical : f.rfilename) === RECOMMENDED_QUANT)
        const keyQuants = new Set(['Q2_K', 'Q4_K_M', 'Q5_K_M', 'Q8_0'])
        const filteredFiles = showAll
          ? files
          : files.filter(f => {
              const q = extractQuant(isShardSet(f) ? f.canonical : f.rfilename)
              return q ? keyQuants.has(q) : false
            })
        const displayFiles = filteredFiles.length >= 2 ? filteredFiles : files
        const isFiltered = displayFiles.length < files.length

        return (
          <div>
            {recommended && !showAll && (
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 'var(--fs-md)', color: 'var(--dim)', marginBottom: 6 }}>Recommended for you:</div>
                <button
                  onClick={() => onSelectFile(recommended)}
                  style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    width: '100%', textAlign: 'left',
                    background: 'color-mix(in srgb, var(--accent) 7%, transparent)',
                    border: '1px solid color-mix(in srgb, var(--accent) 27%, transparent)',
                    borderRadius: 'var(--radius-xl)', padding: '14px 16px', cursor: 'pointer',
                    transition: 'border-color 0.15s',
                  }}
                  onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--accent)')}
                  onMouseLeave={e => (e.currentTarget.style.borderColor = 'color-mix(in srgb, var(--accent) 27%, transparent)')}
                >
                  <div>
                    <div style={{ color: 'var(--accent)', fontSize: 'var(--fs-lg)', fontWeight: 600, fontFamily: 'var(--font-mono)', marginBottom: 4 }}>
                      {isShardSet(recommended) ? recommended.canonical.replace(/-00001-of-\d+\.gguf$/i, '') : recommended.rfilename}
                    </div>
                    <div style={{ fontSize: 'var(--fs-sm)', color: 'var(--dim)' }}>
                      {QUANT_TABLE[RECOMMENDED_QUANT]?.label ?? 'Best balance of size & quality'}
                    </div>
                  </div>
                  <div style={{ textAlign: 'right', flexShrink: 0, marginLeft: 12 }}>
                    <div style={{ color: 'var(--text)', fontSize: 'var(--fs-md)' }}>
                      {(isShardSet(recommended) ? recommended.combinedSize : recommended.size) > 0 ? fmt(isShardSet(recommended) ? recommended.combinedSize : recommended.size) : ''}
                    </div>
                    <div style={{ color: 'var(--accent)', fontSize: 'var(--fs-xs)', marginTop: 2 }}>Select</div>
                  </div>
                </button>
              </div>
            )}

            <div style={{ fontSize: 'var(--fs-md)', color: 'var(--dim)', marginBottom: 6 }}>
              {showAll ? `All variants (${files.length}):` : 'Other variants:'}
            </div>

            {displayFiles.map(f => {
              const filename = isShardSet(f) ? f.canonical : f.rfilename
              const size = isShardSet(f) ? f.combinedSize : f.size
              const quant = extractQuant(filename)
              const info = quant ? getQuantInfo(quant) : null
              const isRecommended = quant === RECOMMENDED_QUANT
              if (!showAll && isRecommended && recommended) return null

              return (
                <button
                  key={filename}
                  onClick={() => onSelectFile(f)}
                  style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    width: '100%', textAlign: 'left',
                    background: 'var(--surface)', border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-lg)', padding: '10px 14px', marginBottom: 4, cursor: 'pointer',
                    transition: 'border-color 0.15s',
                  }}
                  onMouseEnter={e => (e.currentTarget.style.borderColor = 'color-mix(in srgb, var(--accent) 40%, transparent)')}
                  onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--border)')}
                >
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2 }}>
                      <span style={{
                        color: 'var(--text)', fontSize: 'var(--fs-md)', fontFamily: 'var(--font-mono)',
                        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                      }}>
                        {isShardSet(f) ? f.canonical.replace(/-00001-of-\d+\.gguf$/i, '') : f.rfilename}
                      </span>
                      {isShardSet(f) && (
                        <span className={cmp.badge} style={{
                          background: 'color-mix(in srgb, var(--accent) 9%, transparent)',
                          color: 'var(--accent)',
                          border: '1px solid color-mix(in srgb, var(--accent) 20%, transparent)',
                        }}>
                          {f.totalShards} shards
                        </span>
                      )}
                      {info && (
                        <span className={cmp.badge} style={{
                          background: `color-mix(in srgb, ${qualityColor[info.quality]} 9%, transparent)`,
                          color: qualityColor[info.quality],
                          border: `1px solid color-mix(in srgb, ${qualityColor[info.quality]} 20%, transparent)`,
                        }}>
                          {info.quality}
                        </span>
                      )}
                    </div>
                    {info && <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--dim)' }}>{info.label}</div>}
                  </div>
                  <span style={{ color: 'var(--dim)', fontSize: 'var(--fs-sm)', flexShrink: 0, marginLeft: 12 }}>
                    {size > 0 ? fmt(size) : ''}
                  </span>
                </button>
              )
            })}

            {isFiltered && (
              <button
                onClick={onToggleShowAll}
                className={cmp.btnSecondary}
                style={{
                  display: 'block', width: '100%', marginTop: 8,
                  border: '1px dashed var(--border)', borderRadius: 'var(--radius-lg)',
                  padding: '8px', textAlign: 'center',
                }}
              >
                Show all {files.length} variants
              </button>
            )}
            {showAll && files.length > 4 && (
              <button
                onClick={onToggleShowAll}
                className={cmp.btnSecondary}
                style={{
                  display: 'block', width: '100%', marginTop: 8,
                  border: '1px dashed var(--border)', borderRadius: 'var(--radius-lg)',
                  padding: '8px', textAlign: 'center',
                }}
              >
                Show fewer
              </button>
            )}
          </div>
        )
      })()}
    </div>
  )
}
