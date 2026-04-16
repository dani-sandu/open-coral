import React, { useState, useCallback, useEffect, useRef } from 'react'
import type { CoverageReport, InferenceResult, ChatMessage, ChatSession, ModelInfo } from './types'
import CoverageStatus from './CoverageStatus'
import ModelSelector from './ModelSelector'

const C = {
  bg: '#1e1e2e',
  surface: '#181825',
  border: '#313244',
  text: '#cdd6f4',
  dim: '#6c7086',
  accent: '#7c6af7',
  accentDim: '#45396e',
  green: '#a6e3a1',
  red: '#f38ba8',
  yellow: '#f9e2af',
  blue: '#89b4fa',
}

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
    <div style={{
      display: 'flex', alignItems: 'flex-start', marginBottom: 14,
    }}>
      <div style={{
        maxWidth: '88%',
        background: C.surface,
        border: `1px solid ${C.border}`,
        borderRadius: '12px 12px 12px 4px',
        padding: '10px 14px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            fontFamily: 'monospace', fontSize: 14, color: C.accent,
            display: 'inline-block', width: 14, textAlign: 'center',
          }}>
            {SPINNER_FRAMES[frame]}
          </span>
          <span style={{ fontSize: 13, color: C.text }}>
            {THINKING_MESSAGES[msgIdx]}
          </span>
          <span style={{ fontSize: 11, color: C.dim, marginLeft: 8 }}>
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
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '3px 8px',
            marginBottom: 2,
            background: C.bg,
            borderRadius: 4,
            fontSize: 11,
            fontFamily: 'monospace',
            border: `1px solid ${step.peerId === 'local' ? C.accent + '33' : C.border}`,
          }}
        >
          <span style={{ color: C.dim, minWidth: 18, textAlign: 'right' }}>{i + 1}.</span>
          <span style={{ color: step.peerId === 'local' ? C.accent : C.blue, minWidth: 96 }}>
            {shortId(step.peerId)}
          </span>
          <span style={{ color: C.dim }}>
            blk {step.blockStart}–{step.blockEnd}
          </span>
          <span style={{ marginLeft: 'auto', color: C.yellow }}>
            {step.durationMs} ms
          </span>
        </div>
      ))}

      {/* Summary row */}
      <div
        style={{
          display: 'flex',
          gap: 16,
          marginTop: 6,
          padding: '3px 8px',
          fontSize: 11,
          color: C.dim,
          fontFamily: 'monospace',
        }}
      >
        <span>Total: <span style={{ color: C.text }}>{result.totalDurationMs} ms</span></span>
        <span>Generated: <span style={{ color: C.text }}>{result.generatedTokens} tokens</span></span>
      </div>
    </div>
  )
}

function MessageBubble({ msg }: { msg: ChatMessage }): React.JSX.Element {
  const isUser = msg.role === 'user'
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
        marginBottom: 14,
      }}
    >
      <div
        style={{
          maxWidth: '88%',
          background: isUser ? C.accent + '22' : C.surface,
          border: `1px solid ${isUser ? C.accent + '44' : C.border}`,
          borderRadius: isUser ? '12px 12px 4px 12px' : '12px 12px 12px 4px',
          padding: '10px 14px',
        }}
      >
        <div style={{ fontSize: 13, color: C.text, lineHeight: 1.5 }}>
          {msg.text}
        </div>
        {msg.result && <TraceView result={msg.result} />}
        {msg.error && (
          <div style={{ fontSize: 11, color: C.red, marginTop: 6 }}>{msg.error}</div>
        )}
      </div>
      <div style={{ fontSize: 10, color: C.dim, marginTop: 3, padding: '0 4px' }}>
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
  const [maxTokens, setMaxTokens] = useState(32)
  const [sendingSessionId, setSendingSessionId] = useState<string | null>(null)
  const [coverage, setCoverage] = useState<CoverageReport | null>(null)
  const [coverageLoading, setCoverageLoading] = useState(false)
  const [activeModel, setActiveModel] = useState<ModelInfo | null>(null)
  const [showModelPicker, setShowModelPicker] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

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
      const errMsg: ChatMessage = {
        id: `${Date.now()}-error`,
        role: 'assistant',
        text: 'Inference failed.',
        error: String(e),
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

  return (
    <div style={{ display: 'flex', fontFamily: 'system-ui', paddingTop: 16, gap: 0, flex: 1, minHeight: 0 }}>

      {/* ── Session sidebar ──────────────────────────────────────── */}
      <div style={{
        width: 200, flexShrink: 0,
        borderRight: `1px solid ${C.border}`,
        display: 'flex', flexDirection: 'column',
        marginRight: 16,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          <span style={{ fontSize: 11, color: C.dim, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.5 }}>Sessions</span>
          <button
            onClick={onCreateSession}
            style={{
              background: C.accent, color: '#fff', border: 'none',
              borderRadius: 4, width: 22, height: 22, fontSize: 14, lineHeight: '20px',
              cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}
            title="New chat"
          >+</button>
        </div>

        <div style={{ flex: 1, overflowY: 'auto' }}>
          {sessions.length === 0 && (
            <div style={{ color: C.dim, fontSize: 11, padding: '12px 4px', textAlign: 'center' }}>
              No sessions yet.<br />Click + to start.
            </div>
          )}
          {sessions.map(s => (
            <div
              key={s.id}
              style={{
                display: 'flex', alignItems: 'center', gap: 4,
                marginBottom: 2,
              }}
            >
              <button
                onClick={() => onSelectSession(s.id)}
                style={{
                  flex: 1, textAlign: 'left',
                  background: s.id === activeSessionId ? C.accent + '22' : 'transparent',
                  color: s.id === activeSessionId ? C.accent : C.dim,
                  border: s.id === activeSessionId ? `1px solid ${C.accent}44` : '1px solid transparent',
                  borderRadius: 6, padding: '6px 8px', fontSize: 11,
                  cursor: 'pointer', transition: 'all 0.15s',
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}
              >
                <div style={{ fontWeight: 600, marginBottom: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {s.title}
                </div>
                <div style={{ fontSize: 10, opacity: 0.7 }}>
                  {s.messages.length} msgs · {new Date(s.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); onDeleteSession(s.id) }}
                style={{
                  background: 'transparent', color: C.dim, border: 'none',
                  fontSize: 12, cursor: 'pointer', padding: '2px 4px', borderRadius: 4,
                  flexShrink: 0, opacity: 0.5,
                }}
                title="Delete session"
                onMouseEnter={e => (e.currentTarget.style.opacity = '1', e.currentTarget.style.color = C.red)}
                onMouseLeave={e => (e.currentTarget.style.opacity = '0.5', e.currentTarget.style.color = C.dim)}
              >×</button>
            </div>
          ))}
        </div>
      </div>

      {/* ── Main chat area ───────────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <h2 style={{ color: C.text, fontSize: 16, margin: '0 0 12px' }}>
          Chat
        </h2>

        {/* Message history */}
        <div
          ref={scrollRef}
          style={{
            overflowY: 'auto', flex: 1,
            minHeight: 0,
            padding: '4px 0', marginBottom: 4,
          }}
        >
          {!activeSession && (
            <div style={{ textAlign: 'center', color: C.dim, fontSize: 12, padding: '48px 20px', lineHeight: 1.7 }}>
              Select a session from the sidebar or click <strong>+</strong> to start a new chat.
            </div>
          )}
          {activeSession && activeSession.messages.length === 0 && !sending && (
            <div style={{ textAlign: 'center', color: C.dim, fontSize: 12, padding: '32px 20px', lineHeight: 1.7 }}>
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

        {/* Model picker dropdown */}
        {showModelPicker && (
          <div style={{
            background: C.surface, border: `1px solid ${C.border}`,
            borderRadius: 8, padding: '12px 14px', marginBottom: 8,
          }}>
            <ModelSelector currentModel={activeModel} onModelLoaded={handleModelLoaded} />
            {activeModel && (
              <div style={{ marginTop: 10, paddingTop: 10, borderTop: `1px solid ${C.border}` }}>
                <CoverageStatus report={coverage} loading={coverageLoading} onRefresh={checkCoverage} />
              </div>
            )}
          </div>
        )}

        {/* Model name + input area */}
        <div style={{ paddingTop: 12, borderTop: `1px solid ${C.border}` }}>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
            <button
              onClick={() => setShowModelPicker(v => !v)}
              style={{
                background: 'transparent',
                color: activeModel ? C.text : C.dim,
                border: `1px solid ${showModelPicker ? C.accent + '66' : C.border}`,
                borderRadius: 6, padding: '4px 10px', fontSize: 11,
                cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6,
                transition: 'all 0.15s',
                maxWidth: '100%', overflow: 'hidden',
              }}
            >
              <span style={{ color: C.dim, fontSize: 9 }}>{showModelPicker ? '▾' : '▸'}</span>
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {activeModel?.hfFilename
                  ? (activeModel.hfFilename.endsWith('.gguf')
                    ? activeModel.hfFilename.slice(0, -5)
                    : activeModel.hfFilename)
                  : 'No model selected'}
              </span>
              {activeModel && coverage?.complete && (
                <span style={{ color: C.green, fontSize: 9, flexShrink: 0 }}>●</span>
              )}
              {activeModel && coverage && !coverage.complete && (
                <span style={{ color: C.red, fontSize: 9, flexShrink: 0 }}>●</span>
              )}
            </button>
          </div>

          <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
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
              style={{
                flex: 1, background: C.surface, color: C.text,
                border: `1px solid ${C.border}`, borderRadius: 8,
                padding: '8px 12px', fontSize: 13, resize: 'none',
                outline: 'none', fontFamily: 'system-ui',
                opacity: !activeSession || !coverage?.complete ? 0.6 : 1,
              }}
            />

            {/* Token count picker */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3, alignItems: 'center' }}>
              <label style={{ fontSize: 10, color: C.dim }}>max</label>
              <input
                type="number" min={1} max={256}
                value={maxTokens}
                onChange={e => setMaxTokens(Math.max(1, Math.min(256, Number(e.target.value))))}
                style={{
                  width: 44, background: C.surface, color: C.text,
                  border: `1px solid ${C.border}`, borderRadius: 6,
                  padding: '5px 4px', fontSize: 12, textAlign: 'center',
                }}
              />
            </div>

            <button
              onClick={send}
              disabled={!canSend}
              style={{
                background: canSend ? C.accent : C.accentDim,
                color: canSend ? '#fff' : C.dim,
                border: 'none', borderRadius: 8,
                padding: '10px 18px', fontSize: 13,
                cursor: canSend ? 'pointer' : 'default',
                opacity: canSend ? 1 : 0.6,
                alignSelf: 'flex-end',
              }}
            >
              {sending ? '…' : 'Send'}
            </button>
          </div>

          {coverage && !coverage.complete && !coverageLoading && (
            <div style={{ fontSize: 11, color: C.red, marginTop: 6 }}>
              Inference requires all transformer blocks to be covered. Host more blocks or connect peers that cover the missing ranges.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
