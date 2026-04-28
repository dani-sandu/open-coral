import React, { useState, useCallback, useEffect, useRef } from 'react'
import type { CoverageReport, InferenceResult, ChatMessage, ChatSession, ModelInfo, SessionSummary, SessionPhaseEvent } from '../../types'
import CoverageStatus from '../Block/CoverageStatus'
import ModelSelector from '../Model/ModelSelector'
import TabShell from '../shared/TabShell'
import StatusDot from '../shared/StatusDot'
import styles from './ChatPanel.module.css'
import cmp from '../shared/components.module.css'
import { useToast } from '../Toast/ToastProvider'
import { formatMissingRanges } from '../../utils/format'

function shortId(id: string): string {
  if (id === 'local') return 'local'
  return id.length > 12 ? id.slice(0, 8) + '…' : id
}

// ── Thinking spinner (Claude Code style) ────────────────────────────────────

const SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
const THINKING_MESSAGES = [
  'Thinking…',
  'Tokenizing prompt…',
  'Forwarding through blocks…',
  'Running transformer layers…',
  'Passing tensors between peers…',
  'Sampling next token…',
  'Decoding output…',
  'Crunching numbers…',
  'Consulting the weights…',
  'Propagating activations…',
  'Applying attention heads…',
  'Almost there…',
]

function ThinkingIndicator({ phase }: { phase: SessionPhaseEvent | null }): React.JSX.Element {
  const [frame, setFrame] = useState(0)
  const [msgIdx, setMsgIdx] = useState(0)
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    const spinInterval = setInterval(() => {
      setFrame(f => (f + 1) % SPINNER_FRAMES.length)
    }, 80)
    const msgInterval = setInterval(() => {
      setMsgIdx(i => (i + 1) % THINKING_MESSAGES.length)
    }, 3000)
    const elapsedInterval = setInterval(() => {
      setElapsed(e => e + 1)
    }, 1000)
    return () => {
      clearInterval(spinInterval)
      clearInterval(msgInterval)
      clearInterval(elapsedInterval)
    }
  }, [])

  let label: string = THINKING_MESSAGES[msgIdx]
  if (phase) {
    if (phase.phase === 'planning') label = 'Planning chain…'
    else if (phase.phase === 'opening-remote-kv') label = 'Opening remote KV sessions…'
    else if (phase.phase === 'prefilling') {
      const tot = phase.totalTokens ?? 0
      const cur = phase.prefilledTokens
      // We currently emit prefilledTokens only when it's known mid-stream;
      // most paths just emit totalTokens. Show count if granular, size otherwise.
      label = cur !== undefined && tot > 0
        ? `Rebuilding context… (${cur}/${tot} tokens)`
        : tot > 0
          ? `Rebuilding context… (${tot} tokens)`
          : 'Rebuilding context…'
    } else if (phase.phase === 'error') {
      label = `Error: ${phase.error ?? 'unknown'}`
    }
  }

  return (
    <div className={styles.thinkingRow}>
      <div className={styles.thinkingBubble}>
        <div className={styles.thinkingInner}>
          <span className={styles.spinnerFrame}>
            {SPINNER_FRAMES[frame]}
          </span>
          <span className={styles.thinkingMsg}>
            {label}
          </span>
          <span className={styles.thinkingElapsed}>
            {elapsed}s
          </span>
        </div>
      </div>
    </div>
  )
}

function TraceView({ result }: { result: InferenceResult }): React.JSX.Element {
  return (
    <div style={{ marginTop: 10 }}>
      {/* Chain steps */}
      {result.chainSteps.map((step, i) => (
        <div
          key={`${step.peerId}-${step.blockStart}-${i}`}
          className={step.peerId === 'local' ? styles.traceStepLocal : styles.traceStep}
        >
          <span className={styles.traceStepNum}>{i + 1}.</span>
          <span className={step.peerId === 'local' ? styles.traceStepPeerLocal : styles.traceStepPeerRemote}>
            {shortId(step.peerId)}
          </span>
          <span className={styles.traceStepBlocks}>
            blk {step.blockStart}–{step.blockEnd}
          </span>
          <span className={styles.traceStepDuration}>
            {step.durationMs} ms
          </span>
        </div>
      ))}

      {/* Summary row */}
      <div className={styles.traceSummary}>
        <span>Total: <span className={styles.traceSummaryValue}>{result.totalDurationMs} ms</span></span>
        <span>Generated: <span className={styles.traceSummaryValue}>{result.generatedTokens} tokens</span></span>
      </div>
    </div>
  )
}

// ── Thinking block parser ────────────────────────────────────────────────────

function parseThinking(text: string): { thinking: string | null; output: string } {
  // Closed: <think>...</think>output
  const closed = text.match(/^<think>([\s\S]*?)<\/think>([\s\S]*)$/)
  if (closed) return { thinking: closed[1].trim(), output: closed[2].trim() }
  // Still open (model still generating inside <think>)
  const open = text.match(/^<think>([\s\S]*)$/)
  if (open) return { thinking: open[1].trim(), output: '' }
  return { thinking: null, output: text }
}

function ThinkingBlock({ thinking }: { thinking: string }): React.JSX.Element {
  const [open, setOpen] = useState(false)
  return (
    <div className={styles.thinkBlock}>
      <button className={styles.thinkHeader} onClick={() => setOpen(v => !v)}>
        <span className={styles.thinkCaret}>{open ? '▾' : '▸'}</span>
        <span>Thinking</span>
      </button>
      {open && (
        <pre className={styles.thinkContent}>{thinking}</pre>
      )}
    </div>
  )
}

function MessageBubble({ msg }: { msg: ChatMessage }): React.JSX.Element {
  const isUser = msg.role === 'user'
  const { thinking, output } = msg.role === 'assistant' ? parseThinking(msg.text) : { thinking: null, output: msg.text }
  return (
    <div className={isUser ? styles.bubbleUser : styles.bubbleAssistant}>
      <div className={isUser ? styles.bubbleInnerUser : styles.bubbleInnerAssistant}>
        {thinking !== null && thinking !== '' && <ThinkingBlock thinking={thinking} />}
        {output && (
          <div className={styles.bubbleText}>{output}</div>
        )}
        {msg.result && <TraceView result={msg.result} />}
        {msg.error && (
          <div className={styles.bubbleError}>{msg.error}</div>
        )}
      </div>
      <div className={styles.bubbleTimestamp}>
        {new Date(msg.timestamp).toLocaleTimeString()}
      </div>
    </div>
  )
}

interface ChatPanelProps {
  summaries: SessionSummary[]
  activeSessionId: string | null
  onSelectSession: (id: string) => void
  onCreateSession: () => Promise<void>
  onDeleteSession: (id: string) => Promise<void>
}

export default function ChatPanel({
  summaries, activeSessionId, onSelectSession, onCreateSession, onDeleteSession,
}: ChatPanelProps): React.JSX.Element {
  const [draft, setDraft] = useState('')
  const [maxTokens, setMaxTokens] = useState(512)
  const [sendingSessionId, setSendingSessionId] = useState<string | null>(null)
  const [coverage, setCoverage] = useState<CoverageReport | null>(null)
  const [coverageLoading, setCoverageLoading] = useState(false)
  const [activeModel, setActiveModel] = useState<ModelInfo | null>(null)
  const [showModelPicker, setShowModelPicker] = useState(false)
  const [activeSession, setActiveSession] = useState<ChatSession | null>(null)
  const [phase, setPhase] = useState<SessionPhaseEvent | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const { addToast } = useToast()

  // Fetch active session content from main when selection changes.
  useEffect(() => {
    if (!activeSessionId) { setActiveSession(null); return }
    let cancelled = false
    window.opencoral.getSession(activeSessionId).then(s => {
      if (!cancelled) setActiveSession(s)
    })
    return () => { cancelled = true }
  }, [activeSessionId])

  // Subscribe to session-phase events for the rebuilding-context indicator.
  useEffect(() => {
    const off = window.opencoral.onSessionPhase((e) => {
      if (e.sessionId !== activeSessionId) return
      setPhase(e.phase === 'ready' ? null : e)
    })
    return off
  }, [activeSessionId])

  // Refresh active session when it gets updated (auto-title, post-turn refresh).
  useEffect(() => {
    const off = window.opencoral.onSessionUpdated((s) => {
      if (s.id !== activeSessionId) return
      window.opencoral.getSession(s.id).then(fresh => setActiveSession(fresh))
    })
    return off
  }, [activeSessionId])

  const checkCoverage = useCallback(async () => {
    setCoverageLoading(true)
    try {
      const r = await window.opencoral.checkCoverage()
      setCoverage(r)
    } catch {
      setCoverage(null)
    } finally {
      setCoverageLoading(false)
    }
  }, [])

  useEffect(() => {
    if (activeModel) checkCoverage()
  }, [activeModel, checkCoverage])

  const handleModelLoaded = useCallback((model: ModelInfo) => {
    setActiveModel(model)
    setShowModelPicker(false)
    checkCoverage()
  }, [checkCoverage])

  // Scroll to bottom when messages change
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [activeSession?.messages])

  const send = useCallback(async () => {
    const text = draft.trim()
    if (!text || sendingSessionId || coverage?.complete !== true || !activeSession) return

    setDraft('')
    setSendingSessionId(activeSession.id)

    // Optimistically show the user message immediately, before the round-trip.
    setActiveSession(prev => prev ? {
      ...prev,
      messages: [...prev.messages, { id: `optimistic-${Date.now()}`, role: 'user', text, timestamp: Date.now() }],
    } : prev)

    try {
      await window.opencoral.sendTurn(activeSession.id, text, maxTokens)
      // Server-side persisted both messages; the onSessionUpdated subscription
      // above will refetch the session and update the view.
    } catch (e) {
      const errorText = e instanceof Error ? e.message : String(e)
      addToast(errorText, 'error')
      // The user message was already persisted by main; refetch so the UI reflects it.
      const fresh = await window.opencoral.getSession(activeSession.id)
      setActiveSession(fresh)
    } finally {
      setSendingSessionId(null)
      setPhase(null)
    }
  }, [draft, maxTokens, sendingSessionId, coverage, activeSession, addToast])

  const sending = sendingSessionId !== null
  const sendingHere = sendingSessionId !== null && sendingSessionId === activeSessionId
  const canSend = !sending && draft.trim().length > 0 && coverage?.complete === true && activeSession !== null

  const placeholder = !activeSession
    ? 'Create or select a session first…'
    : !coverage
      ? 'Check coverage first…'
      : !coverage.complete
        ? 'All blocks must be covered to chat — connect more peers'
        : 'Type a prompt and press Enter…'

  // Coverage status dot shown in TabShell status strip
  const coverageStatus = activeModel ? (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 'var(--fs-sm)', fontFamily: 'var(--font-mono)' }}>
      {coverageLoading
        ? <span style={{ color: 'var(--dim)' }}>Checking…</span>
        : coverage?.complete
          ? <><StatusDot color="green" /><span style={{ color: 'var(--dim)' }}>All blocks covered</span></>
          : coverage
            ? <><StatusDot color="red" /><span style={{ color: 'var(--dim)' }}>{coverage.totalBlocks - coverage.missing.length}/{coverage.totalBlocks} blocks</span></>
            : <span style={{ color: 'var(--dim)' }}>No coverage data</span>
      }
    </span>
  ) : undefined

  // Model picker toggle shown in TabShell actions strip
  const modelToggle = (
    <button
      onClick={() => setShowModelPicker(v => !v)}
      className={
        showModelPicker
          ? (activeModel ? styles.modelBtnActiveLoaded : styles.modelBtnActive)
          : (activeModel ? styles.modelBtnLoaded : styles.modelBtn)
      }
    >
      <span className={styles.modelBtnCaret}>{showModelPicker ? '▾' : '▸'}</span>
      <span className={styles.modelBtnLabel}>
        {activeModel?.hfFilename
          ? (activeModel.hfFilename.endsWith('.gguf')
            ? activeModel.hfFilename.slice(0, -5)
            : activeModel.hfFilename)
          : 'No model selected'}
      </span>
      {activeModel && coverage?.complete && (
        <StatusDot color="green" />
      )}
      {activeModel && coverage && !coverage.complete && (
        <StatusDot color="red" />
      )}
    </button>
  )

  return (
    <div className={styles.chatLayout}>

      {/* ── Session sidebar ──────────────────────────────────────── */}
      <div className={styles.sidebar}>
        <div className={styles.sidebarHeader}>
          <span className={styles.sidebarLabel}>Sessions</span>
          <button
            onClick={onCreateSession}
            className={cmp.btnIcon}
            title="New chat"
          >+</button>
        </div>

        <div className={styles.sidebarList}>
          {summaries.length === 0 && (
            <div className={styles.sidebarEmpty}>
              No sessions yet.<br />Click + to start.
            </div>
          )}
          {summaries.map(s => (
            <div key={s.id} className={styles.sessionRow}>
              <button
                onClick={() => onSelectSession(s.id)}
                className={s.id === activeSessionId ? styles.sessionBtnActive : styles.sessionBtn}
                style={s.corrupt ? { opacity: 0.5 } : undefined}
                title={s.corrupt ? 'Session file is unreadable' : undefined}
              >
                <div className={styles.sessionBtnTitle}>{s.title}</div>
                <div className={styles.sessionBtnMeta}>
                  {s.messageCount} msgs · {new Date(s.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); onDeleteSession(s.id) }}
                className={styles.deleteBtn}
                title="Delete session"
              >×</button>
            </div>
          ))}
        </div>
      </div>

      {/* ── Main chat area via TabShell ──────────────────────────── */}
      <TabShell title="Chat" status={coverageStatus} actions={modelToggle}>
        <div className={styles.chatMain}>

          {/* Model picker dropdown */}
          {showModelPicker && (
            <div className={styles.modelPickerDropdown}>
              <ModelSelector currentModel={activeModel} onModelLoaded={handleModelLoaded} />
              {activeModel && (
                <div className={styles.modelPickerCoverage}>
                  <CoverageStatus report={coverage} loading={coverageLoading} onRefresh={checkCoverage} />
                </div>
              )}
            </div>
          )}

          {/* Message history */}
          <div ref={scrollRef} className={styles.messageArea}>
            {!activeSession && (
              <div className={styles.messageEmpty}>
                Select a session from the sidebar or click <strong>+</strong> to start a new chat.
              </div>
            )}
            {activeSession && activeSession.messages.length === 0 && !sending && (
              <div className={styles.messageStart}>
                Type a prompt below.<br />
                Your text will be tokenized, forwarded through the transformer<br />
                blocks, and the model's response displayed here.
              </div>
            )}
            {activeSession?.messages.map(msg => (
              <MessageBubble key={msg.id} msg={msg} />
            ))}
            {sendingHere && <ThinkingIndicator phase={phase} />}
          </div>

          {/* Input area */}
          <div className={styles.inputArea}>
            <div className={styles.inputRow}>
              <textarea
                value={draft}
                onChange={e => setDraft(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    send()
                  }
                }}
                placeholder={placeholder}
                disabled={!coverage?.complete || sending || !activeSession}
                rows={2}
                className={styles.textarea}
              />

              {/* Token count picker */}
              <div className={styles.maxTokensGroup}>
                <label className={styles.maxTokensLabel}>max</label>
                <input
                  type="number" min={1} max={2048}
                  value={maxTokens}
                  onChange={e => setMaxTokens(Math.max(1, Math.min(2048, Number(e.target.value))))}
                  className={styles.maxTokensInput}
                />
              </div>

              <button
                onClick={send}
                disabled={!canSend}
                className={cmp.btnPrimary}
                style={{ alignSelf: 'flex-end' }}
              >
                {sending ? '…' : 'Send'}
              </button>
            </div>

            {coverage && !coverage.complete && !coverageLoading && (
              <div className={styles.coverageWarning}>
                {coverage.totalBlocks - coverage.missing.length}/{coverage.totalBlocks} blocks covered.
                {' '}Missing: {formatMissingRanges(coverage.missing)}.
                {' '}Connect peers that host the missing blocks.
              </div>
            )}
          </div>

        </div>
      </TabShell>
    </div>
  )
}
