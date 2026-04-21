import React, { useState, useCallback, useEffect, useRef } from 'react'
import type { CoverageReport, InferenceResult, ChatMessage, ChatSession, ModelInfo } from '../../types'
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

function ThinkingIndicator(): React.JSX.Element {
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

  return (
    <div className={styles.thinkingRow}>
      <div className={styles.thinkingBubble}>
        <div className={styles.thinkingInner}>
          <span className={styles.spinnerFrame}>
            {SPINNER_FRAMES[frame]}
          </span>
          <span className={styles.thinkingMsg}>
            {THINKING_MESSAGES[msgIdx]}
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
  sessions: ChatSession[]
  activeSessionId: string | null
  onSelectSession: (id: string) => void
  onCreateSession: () => void
  onUpdateSession: (id: string, patch: Partial<ChatSession>) => void
  onDeleteSession: (id: string) => void
}

export default function ChatPanel({
  sessions, activeSessionId, onSelectSession, onCreateSession, onUpdateSession, onDeleteSession,
}: ChatPanelProps): React.JSX.Element {
  const [draft, setDraft] = useState('')
  const [maxTokens, setMaxTokens] = useState(512)
  const [sendingSessionId, setSendingSessionId] = useState<string | null>(null)
  const [coverage, setCoverage] = useState<CoverageReport | null>(null)
  const [coverageLoading, setCoverageLoading] = useState(false)
  const [activeModel, setActiveModel] = useState<ModelInfo | null>(null)
  const [showModelPicker, setShowModelPicker] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const { addToast } = useToast()

  const activeSession = sessions.find(s => s.id === activeSessionId) ?? null

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

    const userMsg: ChatMessage = {
      id: `${Date.now()}-user`,
      role: 'user',
      text,
      timestamp: Date.now(),
    }

    // Auto-title from first message
    const isFirst = activeSession.messages.length === 0
    const newMessages = [...activeSession.messages, userMsg]
    const patch: Partial<ChatSession> = { messages: newMessages }
    if (isFirst) {
      patch.title = text.length > 40 ? text.slice(0, 40) + '…' : text
    }
    onUpdateSession(activeSession.id, patch)

    setDraft('')
    setSendingSessionId(activeSession.id)

    try {
      const result = await window.opencoral.runInference(text, maxTokens)
      const assistantMsg: ChatMessage = {
        id: `${Date.now()}-assistant`,
        role: 'assistant',
        text: result.generatedText || '(no output)',
        result,
        timestamp: Date.now(),
      }
      onUpdateSession(activeSession.id, { messages: [...newMessages, assistantMsg] })
    } catch (e) {
      const errorText = e instanceof Error ? e.message : String(e)
      addToast(errorText, 'error')
      const errMsg: ChatMessage = {
        id: `${Date.now()}-error`,
        role: 'assistant',
        text: 'Inference failed.',
        error: errorText,
        timestamp: Date.now(),
      }
      onUpdateSession(activeSession.id, { messages: [...newMessages, errMsg] })
    } finally {
      setSendingSessionId(null)
    }
  }, [draft, maxTokens, sendingSessionId, coverage, activeSession, onUpdateSession])

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
          {sessions.length === 0 && (
            <div className={styles.sidebarEmpty}>
              No sessions yet.<br />Click + to start.
            </div>
          )}
          {sessions.map(s => (
            <div key={s.id} className={styles.sessionRow}>
              <button
                onClick={() => onSelectSession(s.id)}
                className={s.id === activeSessionId ? styles.sessionBtnActive : styles.sessionBtn}
              >
                <div className={styles.sessionBtnTitle}>{s.title}</div>
                <div className={styles.sessionBtnMeta}>
                  {s.messages.length} msgs · {new Date(s.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
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
            {sendingHere && <ThinkingIndicator />}
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
