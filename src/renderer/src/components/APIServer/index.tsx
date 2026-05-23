import React, { useState, useEffect, useCallback, useRef } from 'react'
import TabShell from '../shared/TabShell'
import StatusDot from '../shared/StatusDot'
import cmp from '../shared/components.module.css'
import styles from './APIServer.module.css'

interface ApiServerStatus {
  running: boolean
  port: number
  apiKey: string
  claudeEnabled: boolean
  endpoints: string[]
}

interface LogEntry {
  ts: number
  method: string
  path: string
  status: number
  ms: number
}

const MAX_LOG_ENTRIES = 200

const METHOD_COLOR: Record<string, string> = {
  GET: 'var(--teal)',
  POST: 'var(--accent)',
  PUT: 'var(--yellow)',
  DELETE: 'var(--red)',
}

function logStatusColor(code: number): string {
  if (code < 300) return 'var(--green)'
  if (code < 400) return 'var(--yellow)'
  return 'var(--red)'
}

function formatTime(ts: number): string {
  return new Date(ts).toLocaleTimeString(undefined, { hour12: false })
}

const ENDPOINTS = [
  { method: 'POST', path: '/v1/chat/completions', label: 'Chat Completions' },
  { method: 'POST', path: '/v1/messages',         label: 'Messages' },
  { method: 'GET',  path: '/v1/models',            label: 'Models' },
  { method: 'GET',  path: '/health',               label: 'Health' },
]

export default function APIServerView(): React.JSX.Element {
  const [status, setStatus] = useState<ApiServerStatus | null>(null)
  const [portInput, setPortInput] = useState('')
  const [masked, setMasked] = useState(true)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [copied, setCopied] = useState<string | null>(null)
  const logPaneRef = useRef<HTMLDivElement>(null)
  const atBottomRef = useRef(true)

  const refresh = useCallback(async () => {
    const s = await window.opencoral.apiServerStatus() as ApiServerStatus
    setStatus(s)
    setPortInput(String(s.port))
  }, [])

  useEffect(() => {
    refresh()
    const offStatus = window.opencoral.onApiServerStatusPush((s) => {
      setStatus(s as ApiServerStatus)
      setPortInput(String((s as ApiServerStatus).port))
    })
    const offLog = window.opencoral.onApiServerLog((entry) => {
      setLogs(prev => {
        const next = [...prev, entry as LogEntry]
        return next.length > MAX_LOG_ENTRIES ? next.slice(-MAX_LOG_ENTRIES) : next
      })
    })
    return () => { offStatus(); offLog() }
  }, [refresh])

  // Auto-scroll only when already at the bottom
  useEffect(() => {
    if (!atBottomRef.current) return
    const pane = logPaneRef.current
    if (pane) pane.scrollTop = pane.scrollHeight
  }, [logs])

  const handleLogScroll = useCallback(() => {
    const pane = logPaneRef.current
    if (!pane) return
    atBottomRef.current = pane.scrollHeight - pane.scrollTop - pane.clientHeight < 32
  }, [])

  const toggle = useCallback(async () => {
    if (!status) return
    await window.opencoral.apiServerToggle(!status.running)
  }, [status])

  const savePort = useCallback(async () => {
    const port = parseInt(portInput, 10)
    if (!isNaN(port) && port > 1024 && port < 65535) {
      await window.opencoral.apiServerSetPort(port)
    }
  }, [portInput])

  const regenKey = useCallback(async () => {
    await window.opencoral.apiServerRegenKey()
  }, [])

  const claudeToggle = useCallback(async () => {
    await window.opencoral.apiServerClaudeToggle()
  }, [])

  const copy = useCallback((text: string, id: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(id)
      setTimeout(() => setCopied(id2 => id2 === id ? null : id2), 1500)
    }).catch(() => {})
  }, [])

  if (!status) return <TabShell title="API Server"><div /></TabShell>

  const displayKey = masked
    ? `${status.apiKey.slice(0, 14)}${'•'.repeat(10)}`
    : status.apiKey

  const baseUrl = `http://localhost:${status.port}`

  // ── TabShell header strip ──────────────────────────────────────────────────

  const statusContent = (
    <>
      <StatusDot color={status.running ? 'green' : 'red'} />
      <span style={{ color: 'var(--dim)', fontSize: 'var(--fs-sm)' }}>
        {status.running
          ? <><span style={{ color: 'var(--text)' }}>{baseUrl}</span></>
          : 'Stopped'}
      </span>
    </>
  )

  const actionsContent = (
    <button
      className={status.running ? cmp.btnDanger : cmp.btnSecondary}
      onClick={toggle}
    >
      {status.running ? 'Stop' : 'Start'}
    </button>
  )

  return (
    <TabShell title="API Server" status={statusContent} actions={actionsContent}>
      <div className={styles.root}>

        {/* Configuration card */}
        <div className={styles.card}>
          <span className={styles.cardTitle}>Configuration</span>

          <div className={styles.fieldRow}>
            <span className={styles.fieldLabel}>Port</span>
            <input
              className={`${cmp.input} ${styles.portInput}`}
              value={portInput}
              disabled={status.running}
              onChange={e => setPortInput(e.target.value)}
              onBlur={savePort}
              onKeyDown={e => { if (e.key === 'Enter') savePort() }}
            />
            {status.running && (
              <span className={styles.fieldHint}>Stop server to change</span>
            )}
          </div>

          <div className={styles.fieldRow}>
            <span className={styles.fieldLabel}>API Key</span>
            <div className={styles.keyRow}>
              <span
                className={styles.keyText}
                title={masked ? 'Click to reveal' : 'Click to hide'}
                onClick={() => setMasked(m => !m)}
              >
                {displayKey}
              </span>
              <button
                className={cmp.btnSecondary}
                onClick={() => setMasked(m => !m)}
                title={masked ? 'Reveal' : 'Hide'}
              >
                {masked ? 'Show' : 'Hide'}
              </button>
              <button
                className={`${cmp.btnSecondary} ${copied === 'key' ? styles.copiedFlash : ''}`}
                onClick={() => copy(status.apiKey, 'key')}
              >
                {copied === 'key' ? '✓' : 'Copy'}
              </button>
              <button
                className={cmp.btnSecondary}
                onClick={regenKey}
                title="Regenerate key — takes effect immediately"
              >
                ↻
              </button>
            </div>
          </div>
        </div>

        {/* Endpoints card */}
        <div className={styles.card}>
          <span className={styles.cardTitle}>Endpoints</span>
          <div className={styles.endpointList}>
            {ENDPOINTS.map(ep => {
              const url = `${baseUrl}${ep.path}`
              const copyId = `ep-${ep.path}`
              return (
                <div
                  key={ep.path}
                  className={styles.endpointRow}
                  onClick={() => copy(url, copyId)}
                  title={`Click to copy: ${url}`}
                >
                  <span
                    className={styles.methodBadge}
                    style={{ color: METHOD_COLOR[ep.method] ?? 'var(--dim)', borderColor: METHOD_COLOR[ep.method] ?? 'var(--border)' }}
                  >
                    {ep.method}
                  </span>
                  <span className={styles.endpointPath}>{ep.path}</span>
                  <span className={styles.endpointLabel}>{ep.label}</span>
                  <span className={`${styles.copyHint} ${copied === copyId ? styles.copiedFlash : ''}`}>
                    {copied === copyId ? '✓ Copied' : 'Copy URL'}
                  </span>
                </div>
              )
            })}
          </div>
        </div>

        {/* Claude Code integration */}
        <div className={`${styles.card} ${styles.claudeCard}`}>
          <div className={styles.claudeRow}>
            <div className={styles.claudeInfo}>
              <span className={styles.claudeTitle}>
                Claude Code
                {status.claudeEnabled && <span className={styles.claudeActiveBadge}>active</span>}
              </span>
              <span className={styles.claudeDesc}>
                {status.claudeEnabled
                  ? 'Claude Code is routing through this server. Your original settings are backed up.'
                  : 'Patch ~/.claude/settings.json to route Claude Code through this server.'}
              </span>
            </div>
            <button
              className={status.claudeEnabled ? cmp.btnDanger : cmp.btnSecondary}
              onClick={claudeToggle}
              disabled={!status.running && !status.claudeEnabled}
              title={!status.running && !status.claudeEnabled ? 'Start the server first' : undefined}
            >
              {status.claudeEnabled ? 'Disable' : 'Enable'}
            </button>
          </div>
        </div>

        {/* Access log — fills remaining space */}
        <div className={styles.logCard}>
          <div className={styles.logHeader}>
            <span className={styles.cardTitle}>
              Access Log
              {logs.length > 0 && <span className={styles.logCount}>{logs.length}</span>}
            </span>
            {logs.length > 0 && (
              <button className={cmp.btnSecondary} onClick={() => setLogs([])}>
                Clear
              </button>
            )}
          </div>
          <div
            className={styles.logPane}
            ref={logPaneRef}
            onScroll={handleLogScroll}
          >
            {logs.length === 0
              ? <span className={styles.logEmpty}>No requests yet</span>
              : logs.map((entry, i) => (
                <div key={i} className={styles.logRow}>
                  <span className={styles.logTime}>{formatTime(entry.ts)}</span>
                  <span className={styles.logMethod} style={{ color: METHOD_COLOR[entry.method] ?? 'var(--dim)' }}>
                    {entry.method}
                  </span>
                  <span className={styles.logPath}>{entry.path}</span>
                  <span className={styles.logStatus} style={{ color: logStatusColor(entry.status) }}>
                    {entry.status}
                  </span>
                  <span className={styles.logMs}>{entry.ms}ms</span>
                </div>
              ))
            }
          </div>
        </div>

      </div>
    </TabShell>
  )
}
