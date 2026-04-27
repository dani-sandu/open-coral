import React, { useState, useCallback, useEffect } from 'react'
import NetworkView from './components/Network/NetworkView'
import ModelsPanel from './components/Model/ModelsPanel'
import ChatPanel from './components/Chat/ChatPanel'
import type { SessionSummary } from './types'
import './components/shared/theme.css'
import styles from './App.module.css'
import ToastProvider, { useToast } from './components/Toast/ToastProvider'

type Tab = 'network' | 'models' | 'chat'

const TABS: { id: Tab; label: string }[] = [
  { id: 'network', label: 'Network' },
  { id: 'models', label: 'Models' },
  { id: 'chat', label: 'Chat' },
]

function AppInner(): React.JSX.Element {
  const [tab, setTab] = useState<Tab>('network')
  const [summaries, setSummaries] = useState<SessionSummary[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const { addToast } = useToast()

  useEffect(() => {
    let cancelled = false
    window.opencoral.listSessions().then(list => {
      if (!cancelled) setSummaries(list)
    })
    const offUpdated = window.opencoral.onSessionUpdated((s) => {
      setSummaries(prev => {
        const i = prev.findIndex(x => x.id === s.id)
        if (i === -1) return [s, ...prev]
        const next = prev.slice()
        next[i] = s
        return next
      })
    })
    const offDeleted = window.opencoral.onSessionDeleted((id) => {
      setSummaries(prev => prev.filter(x => x.id !== id))
      setActiveSessionId(prev => prev === id ? null : prev)
    })
    const offInvalidated = window.opencoral.onSessionInvalidated(({ reason }) => {
      addToast(`Session context lost (${reason}) — will rebuild on next message`, 'info')
    })
    return () => { cancelled = true; offUpdated(); offDeleted(); offInvalidated() }
  }, [addToast])

  const createSession = useCallback(async () => {
    const s = await window.opencoral.createSession()
    setActiveSessionId(s.id)
  }, [])

  const deleteSession = useCallback(async (id: string) => {
    await window.opencoral.deleteSession(id)
  }, [])

  return (
    <div className={styles.app}>
      <div className={styles.header}>
        <span className={styles.headerTitle}>OpenCoral</span>
      </div>

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

      <div className={styles.tabContent}>
        <div className={tab === 'network' ? styles.tabPane : styles.tabPaneHidden}>
          <NetworkView />
        </div>
        <div className={tab === 'models' ? styles.tabPane : styles.tabPaneHidden}>
          <ModelsPanel />
        </div>
        <div className={tab === 'chat' ? styles.tabPane : styles.tabPaneHidden}>
          <ChatPanel
            summaries={summaries}
            activeSessionId={activeSessionId}
            onSelectSession={setActiveSessionId}
            onCreateSession={createSession}
            onDeleteSession={deleteSession}
          />
        </div>
      </div>
    </div>
  )
}

export default function App(): React.JSX.Element {
  return (
    <ToastProvider>
      <AppInner />
    </ToastProvider>
  )
}
