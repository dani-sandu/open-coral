import React, { useState } from 'react'
import NetworkView from './NetworkView'
import ModelPanel from './ModelPanel'
import BlockHostPanel from './BlockHostPanel'
import ChatPanel from './ChatPanel'
import './types'

// ── Shared color palette ───────────────────────────────────────────────────────

const C = {
  bg: '#1e1e2e', surface: '#181825', border: '#313244',
  text: '#cdd6f4', dim: '#6c7086', accent: '#7c6af7',
}

type Tab = 'network' | 'model' | 'blocks' | 'chat'

const TABS: { id: Tab; label: string }[] = [
  { id: 'network', label: 'Network' },
  { id: 'model', label: 'Model' },
  { id: 'blocks', label: 'Blocks' },
  { id: 'chat', label: 'Chat' },
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
        {tab === 'chat' && <ChatPanel />}
      </div>
    </div>
  )
}
