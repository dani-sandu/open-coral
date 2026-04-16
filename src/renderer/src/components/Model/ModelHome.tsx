import React from 'react'
import type {
  LocalModelEntry, ModelInfo, HostingState,
} from '../../types'
import { fmt } from '../shared/format-utils'
import cmp from '../shared/components.module.css'
import StatusDot from '../shared/StatusDot'

// ── Quantization helpers ─────────────────────────────────────────────────────

interface QuantInfo {
  quant: string
  quality: 'low' | 'medium' | 'high'
  label: string
  order: number
}

const QUANT_TABLE: Record<string, QuantInfo> = {
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

function extractQuant(filename: string): string | null {
  const m = filename.match(/[.-]((?:IQ|Q|F|BF)\d+(?:_K)?(?:_[A-Z]+)?|F16|F32|BF16)(?:[.-]|$)/i)
  return m ? m[1].toUpperCase() : null
}

function getQuantInfo(quant: string): QuantInfo {
  return QUANT_TABLE[quant] ?? { quant, quality: 'medium' as const, label: quant, order: 50 }
}

const qualityColor: Record<string, string> = {
  low: 'var(--yellow)', medium: 'var(--green)', high: 'var(--blue)',
}

// ── Props ────────────────────────────────────────────────────────────────────

export interface ModelHomeProps {
  localModels: LocalModelEntry[]
  selectedModel: LocalModelEntry | null
  activeModel: ModelInfo | null
  hostingState: HostingState | null
  hostBlockStart: number
  hostBlockEnd: number
  loadingModels: boolean
  hostBusy: boolean
  error: string | null
  query: string
  searching: boolean
  loading: boolean
  onSelectModel: (entry: LocalModelEntry | null) => void
  onSetHostBlockStart: (v: number) => void
  onSetHostBlockEnd: (v: number) => void
  onStartHosting: () => void
  onStopHosting: () => void
  onSetQuery: (q: string) => void
  onSearch: () => void
  onPickLocal: () => void
}

// ── Component ────────────────────────────────────────────────────────────────

export default function ModelHome({
  localModels, selectedModel, activeModel, hostingState,
  hostBlockStart, hostBlockEnd, loadingModels, hostBusy,
  error, query, searching, loading,
  onSelectModel, onSetHostBlockStart, onSetHostBlockEnd,
  onStartHosting, onStopHosting, onSetQuery, onSearch, onPickLocal,
}: ModelHomeProps): React.JSX.Element {

  const isHostingModel = (entry: LocalModelEntry) =>
    hostingState != null && hostingState.modelPath === entry.path

  const maxHostBlock = selectedModel
    ? (selectedModel.blockEnd ?? selectedModel.totalBlocks - 1)
    : 0
  const minHostBlock = selectedModel
    ? (selectedModel.blockStart ?? 0)
    : 0

  return (
    <>
      {/* ── Search / Add model (always visible at top) ──────────── */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
          <input
            type="text"
            value={query}
            onChange={e => onSetQuery(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') onSearch() }}
            placeholder="Search Hugging Face GGUF models..."
            className={cmp.input}
            style={{
              flex: 1, borderRadius: 'var(--radius-lg)',
              padding: '10px 14px', fontSize: 'var(--fs-lg)',
            }}
          />
          <button
            onClick={onSearch}
            disabled={searching || !query.trim()}
            className={cmp.btnPrimary}
            style={{ borderRadius: 'var(--radius-lg)' }}
          >
            {searching ? 'Searching...' : 'Search HF'}
          </button>
          <button
            onClick={onPickLocal}
            disabled={loading}
            className={cmp.btnSecondary}
            style={{ borderRadius: 'var(--radius-lg)', padding: '10px 14px', fontSize: 'var(--fs-lg)' }}
          >
            {loading ? 'Loading...' : 'Local file'}
          </button>
        </div>
      </div>

      {/* ── Currently Hosting ───────────────────────────────────── */}
      {hostingState && (() => {
        const hostedEntry = localModels.find(e => e.path === hostingState.modelPath)
        return (
          <div style={{
            background: 'color-mix(in srgb, var(--green) 5%, transparent)',
            border: '1px solid color-mix(in srgb, var(--green) 20%, transparent)',
            borderRadius: 'var(--radius-xl)', padding: 14, marginBottom: 20,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <StatusDot color="green" />
                <span style={{ color: 'var(--green)', fontSize: 'var(--fs-lg)', fontWeight: 600 }}>Currently Hosting</span>
              </div>
              <button
                onClick={onStopHosting}
                disabled={hostBusy}
                className={cmp.btnDanger}
                style={{ padding: '4px 12px', fontSize: 'var(--fs-sm)' }}
              >
                {hostBusy ? 'Stopping...' : 'Stop'}
              </button>
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--fs-md)' }}>
              <div style={{ color: 'var(--text)', fontWeight: 600, marginBottom: 6 }}>
                {hostedEntry?.filename ?? hostingState.modelPath.split(/[\\/]/).pop()}
              </div>
              <div style={{ display: 'flex', gap: 16, color: 'var(--dim)', fontSize: 'var(--fs-sm)' }}>
                <span>Blocks {hostingState.blockStart}&ndash;{hostingState.blockEnd}</span>
                <span>Hidden {hostingState.hiddenSize}</span>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 2, marginTop: 10 }}>
              {Array.from({ length: hostingState.totalBlocks }, (_, i) => {
                const hosted = i >= hostingState.blockStart && i <= hostingState.blockEnd
                return (
                  <div
                    key={i}
                    style={{
                      flex: 1, height: 10, borderRadius: 2,
                      background: hosted ? 'var(--green)' : 'var(--border)',
                      opacity: hosted ? 1 : 0.25,
                    }}
                    title={`Block ${i}${hosted ? ' (hosted)' : ''}`}
                  />
                )
              })}
            </div>
          </div>
        )
      })()}

      {/* ── Loading spinner ─────────────────────────────────────── */}
      {loadingModels && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 32, gap: 10 }}>
          <svg width="20" height="20" viewBox="0 0 20 20" style={{ animation: 'spin 1s linear infinite' }}>
            <circle cx="10" cy="10" r="8" fill="none" stroke="var(--accent)" strokeWidth="2" strokeDasharray="40 20" strokeLinecap="round" />
          </svg>
          <span style={{ color: 'var(--dim)', fontSize: 'var(--fs-md)' }}>Loading models...</span>
          <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
        </div>
      )}

      {/* ── Downloaded models list (cards) ──────────────────────── */}
      {!loadingModels && localModels.length > 0 && (
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 'var(--fs-sm)', color: 'var(--dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.5, fontWeight: 600 }}>
            Downloaded models
          </div>
          {localModels.map(entry => {
            const hosting = isHostingModel(entry)
            const isSelected = selectedModel?.path === entry.path
            const rangeLabel = entry.blockStart != null && entry.blockEnd != null
              ? `blocks ${entry.blockStart}\u2013${entry.blockEnd} of ${entry.totalBlocks}`
              : `all ${entry.totalBlocks} blocks`
            const quant = extractQuant(entry.filename)

            return (
              <div key={entry.path} style={{ marginBottom: 6 }}>
                {/* Model card header */}
                <button
                  onClick={() => onSelectModel(isSelected ? null : entry)}
                  style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    width: '100%', textAlign: 'left',
                    background: isSelected
                      ? 'color-mix(in srgb, var(--accent) 8%, transparent)'
                      : 'var(--surface)',
                    border: isSelected
                      ? '1px solid color-mix(in srgb, var(--accent) 33%, transparent)'
                      : '1px solid var(--border)',
                    borderRadius: isSelected ? '10px 10px 0 0' : 'var(--radius-xl)',
                    padding: '12px 14px', cursor: 'pointer',
                    transition: 'border-color 0.15s, background 0.15s',
                    fontFamily: 'var(--font-ui)',
                  }}
                  onMouseEnter={e => { if (!isSelected) e.currentTarget.style.borderColor = 'color-mix(in srgb, var(--accent) 27%, transparent)' }}
                  onMouseLeave={e => { if (!isSelected) e.currentTarget.style.borderColor = 'var(--border)' }}
                >
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                      <span style={{
                        color: 'var(--text)', fontSize: 'var(--fs-lg)', fontWeight: 600, fontFamily: 'var(--font-mono)',
                        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                      }}>
                        {entry.filename}
                      </span>
                      {quant && (() => {
                        const info = getQuantInfo(quant)
                        return (
                          <span className={cmp.badge} style={{
                            background: `color-mix(in srgb, ${qualityColor[info.quality]} 9%, transparent)`,
                            color: qualityColor[info.quality],
                            border: `1px solid color-mix(in srgb, ${qualityColor[info.quality]} 20%, transparent)`,
                          }}>
                            {info.quality}
                          </span>
                        )
                      })()}
                    </div>
                    <div style={{ fontSize: 'var(--fs-sm)', color: 'var(--dim)' }}>
                      {entry.architecture} &middot; {rangeLabel} &middot; {fmt(entry.fileSizeBytes)}
                    </div>
                  </div>
                  <div style={{ flexShrink: 0, marginLeft: 12, textAlign: 'right', display: 'flex', alignItems: 'center', gap: 8 }}>
                    {hosting ? (
                      <span style={{
                        fontSize: 'var(--fs-xs)', padding: '3px 8px', borderRadius: 'var(--radius-sm)',
                        background: 'color-mix(in srgb, var(--green) 9%, transparent)',
                        color: 'var(--green)',
                        border: '1px solid color-mix(in srgb, var(--green) 20%, transparent)',
                      }}>
                        Hosting {hostingState!.blockStart}&ndash;{hostingState!.blockEnd}
                      </span>
                    ) : (
                      <span style={{ color: 'var(--dim)', fontSize: 'var(--fs-xs)' }}>{isSelected ? 'Collapse' : 'Configure'}</span>
                    )}
                    <span style={{ color: 'var(--dim)', fontSize: 'var(--fs-xs)', transition: 'transform 0.2s', transform: isSelected ? 'rotate(180deg)' : 'rotate(0deg)' }}>{'\u25BE'}</span>
                  </div>
                </button>

                {/* Expanded inline detail */}
                {isSelected && selectedModel && (
                  <div style={{
                    background: 'var(--surface)',
                    border: '1px solid color-mix(in srgb, var(--accent) 33%, transparent)',
                    borderTop: 'none',
                    borderRadius: '0 0 10px 10px', padding: 16,
                  }}>
                    {/* Model metadata */}
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--fs-md)', marginBottom: 14 }}>
                      {([
                        ['Architecture', selectedModel.architecture],
                        ['Total blocks', selectedModel.totalBlocks],
                        ['Hidden size', selectedModel.hiddenSize],
                        ['File size', fmt(selectedModel.fileSizeBytes)],
                        ['Downloaded range', selectedModel.blockStart != null
                          ? `${selectedModel.blockStart}\u2013${selectedModel.blockEnd}`
                          : 'Full model'],
                      ] as [string, string | number][]).map(([l, v]) => (
                        <div key={l} style={{
                          display: 'flex', justifyContent: 'space-between',
                          padding: '3px 0', borderBottom: '1px solid var(--border)',
                        }}>
                          <span style={{ color: 'var(--dim)' }}>{l}</span>
                          <span style={{ color: 'var(--text)' }}>{v}</span>
                        </div>
                      ))}
                    </div>

                    {/* Block selector to host */}
                    {!hosting && (
                      <>
                        <div style={{ fontSize: 'var(--fs-md)', color: 'var(--text)', fontWeight: 600, marginBottom: 8 }}>
                          Select blocks to host
                        </div>
                        <div style={{ fontSize: 'var(--fs-sm)', color: 'var(--dim)', marginBottom: 10 }}>
                          Choose which downloaded blocks to serve on the network.
                        </div>

                        <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 10 }}>
                          <label style={{ color: 'var(--dim)', fontSize: 'var(--fs-md)' }}>Start</label>
                          <input
                            type="number"
                            className={cmp.inputNumber}
                            min={minHostBlock}
                            max={maxHostBlock}
                            value={hostBlockStart}
                            onChange={e => {
                              const v = Math.max(minHostBlock, Math.min(Number(e.target.value), maxHostBlock))
                              onSetHostBlockStart(v)
                              if (hostBlockEnd < v) onSetHostBlockEnd(v)
                            }}
                          />
                          <label style={{ color: 'var(--dim)', fontSize: 'var(--fs-md)' }}>End</label>
                          <input
                            type="number"
                            className={cmp.inputNumber}
                            min={hostBlockStart}
                            max={maxHostBlock}
                            value={hostBlockEnd}
                            onChange={e => {
                              const v = Math.max(hostBlockStart, Math.min(Number(e.target.value), maxHostBlock))
                              onSetHostBlockEnd(v)
                            }}
                          />
                          <span style={{ color: 'var(--dim)', fontSize: 'var(--fs-sm)' }}>
                            {hostBlockEnd - hostBlockStart + 1} of {selectedModel.totalBlocks} blocks
                          </span>
                        </div>

                        <div style={{ display: 'flex', gap: 2, marginBottom: 14 }}>
                          {Array.from({ length: selectedModel.totalBlocks }, (_, i) => {
                            const downloaded = selectedModel.blockStart != null
                              ? i >= selectedModel.blockStart && i <= (selectedModel.blockEnd ?? selectedModel.totalBlocks - 1)
                              : true
                            const selected = i >= hostBlockStart && i <= hostBlockEnd
                            let bg = 'var(--border)'
                            let opacity = 0.25
                            if (downloaded && selected) {
                              bg = 'var(--accent)'
                              opacity = 1
                            } else if (downloaded) {
                              bg = 'var(--dim)'
                              opacity = 0.45
                            }
                            return (
                              <div
                                key={i}
                                style={{
                                  flex: 1, height: 16, borderRadius: 2,
                                  background: bg, opacity,
                                  transition: 'background 0.15s, opacity 0.15s',
                                  cursor: downloaded ? 'pointer' : 'default',
                                }}
                                title={`Block ${i}${!downloaded ? ' (not downloaded)' : selected ? ' (selected)' : ''}`}
                              />
                            )
                          })}
                        </div>

                        <div style={{ display: 'flex', gap: 6, fontSize: 'var(--fs-xs)', color: 'var(--dim)', marginBottom: 14 }}>
                          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                            <span style={{ width: 8, height: 8, borderRadius: 2, background: 'var(--accent)', display: 'inline-block' }} />
                            Selected to host
                          </span>
                          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                            <span style={{ width: 8, height: 8, borderRadius: 2, background: 'var(--dim)', opacity: 0.45, display: 'inline-block' }} />
                            Downloaded
                          </span>
                          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                            <span style={{ width: 8, height: 8, borderRadius: 2, background: 'var(--border)', opacity: 0.25, display: 'inline-block' }} />
                            Not downloaded
                          </span>
                        </div>

                        <button
                          onClick={onStartHosting}
                          disabled={hostBusy}
                          className={cmp.btnPrimary}
                          style={{ borderRadius: 'var(--radius-lg)' }}
                        >
                          {hostBusy ? 'Starting...' : `Start Hosting ${hostBlockEnd - hostBlockStart + 1} blocks`}
                        </button>
                      </>
                    )}

                    {/* Currently hosting — stop button */}
                    {hosting && hostingState && (
                      <div>
                        <div style={{ fontSize: 'var(--fs-md)', color: 'var(--text)', fontWeight: 600, marginBottom: 8 }}>
                          Hosted blocks
                        </div>
                        <div style={{ display: 'flex', gap: 2, marginBottom: 14 }}>
                          {Array.from({ length: selectedModel.totalBlocks }, (_, i) => {
                            const hosted = i >= hostingState.blockStart && i <= hostingState.blockEnd
                            return (
                              <div
                                key={i}
                                style={{
                                  flex: 1, height: 16, borderRadius: 2,
                                  background: hosted ? 'var(--green)' : 'var(--border)',
                                  opacity: hosted ? 1 : 0.25,
                                  transition: 'background 0.15s, opacity 0.15s',
                                }}
                                title={`Block ${i}${hosted ? ' (hosted)' : ''}`}
                              />
                            )
                          })}
                        </div>
                        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--fs-md)', marginBottom: 12 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', color: 'var(--dim)' }}>
                            <span>Blocks</span>
                            <span style={{ color: 'var(--text)' }}>{hostingState.blockStart}&ndash;{hostingState.blockEnd}</span>
                          </div>
                        </div>
                        <button
                          onClick={onStopHosting}
                          disabled={hostBusy}
                          className={cmp.btnDanger}
                        >
                          {hostBusy ? 'Stopping...' : 'Stop Hosting'}
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

      {/* ── Empty state ─────────────────────────────────────────── */}
      {!loadingModels && localModels.length === 0 && (
        <div style={{ textAlign: 'center', padding: '32px 20px', color: 'var(--dim)', fontSize: 'var(--fs-md)', lineHeight: 1.7 }}>
          No models downloaded yet. Search Hugging Face above or select a local GGUF file.
        </div>
      )}
    </>
  )
}
