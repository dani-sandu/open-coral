import React, { useState } from 'react'
import NetworkView from './NetworkView'
import ModelPanel from './ModelPanel'
import BlockHostPanel from './BlockHostPanel'
import InferencePanel from './InferencePanel'

// ── Types shared between main and renderer ─────────────────────────────────────

interface ModelInfo {
  path: string; architecture: string; totalBlocks: number
  hiddenSize: number; headCount: number; fileSizeBytes: number
}
interface HostingState {
  modelPath: string; blockStart: number; blockEnd: number
  totalBlocks: number; hiddenSize: number
}
interface InferenceDemoResult {
  nTokens: number; nEmbd: number
  chainSteps: { peerId: string; blockStart: number; blockEnd: number; durationMs: number }[]
  totalDurationMs: number; outputNorm: number
}
interface NetworkState {
  localPeerId: string; localMultiaddrs: string[]
  localBlocks: { start: number; end: number }[]
  peers: { peerId: string; multiaddrs: string[]; blockRanges: { start: number; end: number }[]; isLocal: boolean; connected: boolean }[]
  connections: { from: string; to: string }[]
  timestamp: number
}

declare global {
  interface Window {
    coral: {
      version: () => string
      getNetworkState: () => Promise<NetworkState | null>
      selectModel: () => Promise<ModelInfo | null>
      loadModelPath: (filePath: string) => Promise<ModelInfo>
      getModel: () => Promise<ModelInfo | null>
      startHosting: (blockStart: number, blockEnd: number) => Promise<void>
      stopHosting: () => Promise<void>
      getHostingState: () => Promise<HostingState | null>
      runInferenceDemo: (nTokens: number) => Promise<InferenceDemoResult>
      hfSearch: (query: string) => Promise<HFModelResult[]>
      hfListFiles: (repoId: string) => Promise<HFFileInfo[]>
      hfDownload: (repoId: string, filename: string) => Promise<string>
      hfDownloadProgress: () => Promise<DownloadProgress | null>
      hfCancelDownload: () => Promise<void>
      hfPreviewModel: (repoId: string, filename: string) => Promise<HFModelPreview>
      hfEstimateBlocks: (blockStart: number, blockEnd: number) => Promise<BlockEstimate>
      hfDownloadPartial: (repoId: string, filename: string, blockStart: number, blockEnd: number) => Promise<string>
    }
  }
}

interface HFModelResult {
  id: string; author: string; modelId: string
  likes: number; downloads: number; tags: string[]; lastModified: string
}
interface HFFileInfo { rfilename: string; size: number }
interface DownloadProgress {
  file: string; downloadedBytes: number; totalBytes: number
  percent: number; done: boolean; error?: string; localPath?: string
}
interface HFModelPreview {
  architecture: string; totalBlocks: number; hiddenSize: number
  headCount: number; fullFileSize: number; repoId: string; filename: string
}
interface BlockEstimate {
  partialSize: number; fullSize: number; savedPercent: number
}

// ── Shared color palette ───────────────────────────────────────────────────────

const C = {
  bg: '#1e1e2e', surface: '#181825', border: '#313244',
  text: '#cdd6f4', dim: '#6c7086', accent: '#7c6af7',
}

type Tab = 'network' | 'model' | 'blocks' | 'inference'

const TABS: { id: Tab; label: string }[] = [
  { id: 'network', label: 'Network' },
  { id: 'model', label: 'Model' },
  { id: 'blocks', label: 'Blocks' },
  { id: 'inference', label: 'Inference' },
]

export default function App(): React.JSX.Element {
  const [tab, setTab] = useState<Tab>('network')

  return (
    <div style={{
      fontFamily: 'system-ui', background: C.bg, color: C.text,
      minHeight: '100vh', display: 'flex', flexDirection: 'column',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 12,
        padding: '12px 24px', background: C.surface,
        borderBottom: `1px solid ${C.border}`,
      }}>
        <span style={{ color: C.accent, fontSize: 20, fontWeight: 700 }}>⬡</span>
        <span style={{ fontWeight: 700, fontSize: 16, color: C.text }}>Coral</span>
        <span style={{ color: C.dim, fontSize: 11 }}>Decentralized LLM</span>
      </div>

      {/* Tab bar */}
      <div style={{
        display: 'flex', gap: 2, padding: '8px 16px',
        background: C.surface, borderBottom: `1px solid ${C.border}`,
      }}>
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              background: tab === t.id ? C.accent + '22' : 'transparent',
              color: tab === t.id ? C.accent : C.dim,
              border: tab === t.id ? `1px solid ${C.accent}44` : '1px solid transparent',
              borderRadius: 6, padding: '5px 14px', fontSize: 12,
              cursor: 'pointer', transition: 'all 0.15s',
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div style={{ flex: 1, padding: '4px 16px 16px' }}>
        {tab === 'network' && <NetworkView />}
        {tab === 'model' && <ModelPanel />}
        {tab === 'blocks' && <BlockHostPanel />}
        {tab === 'inference' && <InferencePanel />}
      </div>
    </div>
  )
}
