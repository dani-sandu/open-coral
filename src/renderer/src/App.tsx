import React, { useState, useCallback } from 'react'
import NetworkView from './NetworkView'
import ModelsPanel from './ModelsPanel'
import ChatPanel from './ChatPanel'
import type { ChatSession } from './types'
import './types'

// ── Shared color palette ───────────────────────────────────────────────────────

const C = {
  bg: '#1e1e2e', surface: '#181825', border: '#313244',
  text: '#cdd6f4', dim: '#6c7086', accent: '#7c6af7',
}

type Tab = 'network' | 'models' | 'chat'

const TABS: { id: Tab; label: string }[] = [
  { id: 'network', label: 'Network' },
  { id: 'models', label: 'Models' },
  { id: 'chat', label: 'Chat' },
]

export default function App(): React.JSX.Element {
  const [tab, setTab] = useState<Tab>('network')

  // ── Chat session state (persists across tab switches) ─────────────────────
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)

  const createSession = useCallback(() => {
    const s: ChatSession = {
      id: `session-${Date.now()}`,
      title: 'New chat',
      messages: [],
      createdAt: Date.now(),
    }
    setSessions(prev => [s, ...prev])
    setActiveSessionId(s.id)
  }, [])

  const updateSession = useCallback((id: string, patch: Partial<ChatSession>) => {
    setSessions(prev => prev.map(s => s.id === id ? { ...s, ...patch } : s))
  }, [])

  const deleteSession = useCallback((id: string) => {
    setSessions(prev => prev.filter(s => s.id !== id))
    setActiveSessionId(prev => prev === id ? null : prev)
  }, [])

  return (
    <div style={{
      fontFamily: 'system-ui', background: C.bg, color: C.text,
      height: '100vh', display: 'flex', flexDirection: 'column', overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 12,
        padding: '12px 24px', background: C.surface,
        borderBottom: `1px solid ${C.border}`,
      }}>
        <span style={{ color: C.accent, fontSize: 20, fontWeight: 700 }}>⬡</span>
        <span style={{ fontWeight: 700, fontSize: 16, color: C.text }}>OpenCoral</span>
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

      {/* Tab content — all tabs stay mounted, hidden via display:none to preserve state */}
      <div style={{ flex: 1, padding: '4px 16px 16px', display: 'flex', flexDirection: 'column', overflow: 'hidden', minHeight: 0 }}>
        <div style={{ flex: 1, display: tab === 'network' ? 'flex' : 'none', flexDirection: 'column', minHeight: 0 }}>
          <NetworkView />
        </div>
        <div style={{ flex: 1, display: tab === 'models' ? 'flex' : 'none', flexDirection: 'column', minHeight: 0 }}>
          <ModelsPanel />
        </div>
        <div style={{ flex: 1, display: tab === 'chat' ? 'flex' : 'none', flexDirection: 'column', minHeight: 0 }}>
          <ChatPanel
            sessions={sessions}
            activeSessionId={activeSessionId}
            onSelectSession={setActiveSessionId}
            onCreateSession={createSession}
            onUpdateSession={updateSession}
            onDeleteSession={deleteSession}
          />
        </div>
      </div>
    </div>
  )
}
