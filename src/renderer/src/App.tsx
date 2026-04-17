import React, { useState, useCallback } from 'react'
import NetworkView from './components/Network/NetworkView'
import ModelsPanel from './components/Model/ModelsPanel'
import ChatPanel from './components/Chat/ChatPanel'
import type { ChatSession } from './types'
import './components/shared/theme.css'
import styles from './App.module.css'
import ToastProvider from './components/Toast/ToastProvider'

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
    <ToastProvider>
      <div className={styles.app}>
      {/* Header */}
      <div className={styles.header}>
        <span className={styles.headerTitle}>OpenCoral</span>
      </div>

      {/* Tab bar */}
      <div className={styles.tabBar}>
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={tab === t.id ? styles.tabActive : styles.tab}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content — all tabs stay mounted, hidden via display:none to preserve state */}
      <div className={styles.tabContent}>
        <div className={tab === 'network' ? styles.tabPane : styles.tabPaneHidden}>
          <NetworkView />
        </div>
        <div className={tab === 'models' ? styles.tabPane : styles.tabPaneHidden}>
          <ModelsPanel />
        </div>
        <div className={tab === 'chat' ? styles.tabPane : styles.tabPaneHidden}>
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
    </ToastProvider>
  )
}
