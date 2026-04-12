import React, { useState, useCallback, useEffect, useRef } from 'react'
import type { CoverageReport, InferenceResult, ChatMessage, ModelInfo } from './types'
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

export default function ChatPanel(): React.JSX.Element {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [draft, setDraft] = useState('')
  const [maxTokens, setMaxTokens] = useState(32)
  const [sending, setSending] = useState(false)
  const [coverage, setCoverage] = useState<CoverageReport | null>(null)
  const [coverageLoading, setCoverageLoading] = useState(false)
  const [activeModel, setActiveModel] = useState<ModelInfo | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  const checkCoverage = useCallback(async () => {
    setCoverageLoading(true)
    try {
      const r = await window.coral.checkCoverage()
      setCoverage(r)
    } catch {
      setCoverage(null)
    } finally {
      setCoverageLoading(false)
    }
  }, [])

  // Load current model and check coverage on mount
  useEffect(() => {
    window.coral.getModel().then(m => { if (m) setActiveModel(m) })
    checkCoverage()
  }, [checkCoverage])

  const handleModelLoaded = useCallback((model: ModelInfo) => {
    setActiveModel(model)
    checkCoverage()
  }, [checkCoverage])

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  const send = useCallback(async () => {
    const text = draft.trim()
    if (!text || sending || coverage?.complete !== true) return

    const userMsg: ChatMessage = {
      id: `${Date.now()}-user`,
      role: 'user',
      text,
      timestamp: Date.now(),
    }
    setMessages(prev => [...prev, userMsg])
    setDraft('')
    setSending(true)

    try {
      const result = await window.coral.runInference(text, maxTokens)

      const assistantMsg: ChatMessage = {
        id: `${Date.now()}-assistant`,
        role: 'assistant',
        text: result.generatedText || '(no output)',
        result,
        timestamp: Date.now(),
      }
      setMessages(prev => [...prev, assistantMsg])
    } catch (e) {
      const errMsg: ChatMessage = {
        id: `${Date.now()}-error`,
        role: 'assistant',
        text: 'Inference failed.',
        error: String(e),
        timestamp: Date.now(),
      }
      setMessages(prev => [...prev, errMsg])
    } finally {
      setSending(false)
    }
  }, [draft, maxTokens, sending, coverage])

  const canSend = !sending && draft.trim().length > 0 && coverage?.complete === true

  const placeholder = !coverage
    ? 'Check coverage first…'
    : !coverage.complete
      ? 'All blocks must be covered to chat — connect more peers'
      : 'Type a prompt and press Enter…'

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        fontFamily: 'system-ui',
        paddingTop: 16,
      }}
    >
      <h2 style={{ color: C.text, fontSize: 16, margin: '0 0 12px' }}>
        <span style={{ color: C.accent }}>⬡</span> Chat
      </h2>

      {/* Model selector */}
      <div
        style={{
          padding: '12px 16px 0',
          borderBottom: `1px solid ${C.border}`,
          marginBottom: 12,
          paddingBottom: 12,
        }}
      >
        <ModelSelector
          currentModel={activeModel}
          onModelLoaded={handleModelLoaded}
        />
      </div>

      {/* Coverage panel */}
      <div
        style={{
          background: C.surface,
          border: `1px solid ${C.border}`,
          borderRadius: 8,
          padding: '10px 14px',
          marginBottom: 12,
        }}
      >
        <CoverageStatus
          report={coverage}
          loading={coverageLoading}
          onRefresh={checkCoverage}
        />
      </div>

      {/* Message history */}
      <div
        ref={scrollRef}
        style={{
          overflowY: 'auto',
          minHeight: 180,
          maxHeight: 420,
          padding: '4px 0',
          marginBottom: 4,
        }}
      >
        {messages.length === 0 && !sending && (
          <div
            style={{
              textAlign: 'center',
              color: C.dim,
              fontSize: 12,
              padding: '32px 20px',
              lineHeight: 1.7,
            }}
          >
            Type a prompt below.<br />
            Your text will be tokenized, forwarded through the transformer<br />
            blocks, and the model's response displayed here.
          </div>
        )}
        {messages.map(msg => (
          <MessageBubble key={msg.id} msg={msg} />
        ))}
        {sending && (
          <div style={{ color: C.dim, fontSize: 12, padding: '4px 0' }}>
            Running inference…
          </div>
        )}
      </div>

      {/* Input area */}
      <div
        style={{
          display: 'flex',
          gap: 8,
          alignItems: 'flex-end',
          paddingTop: 12,
          borderTop: `1px solid ${C.border}`,
        }}
      >
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
          disabled={!coverage?.complete || sending}
          rows={2}
          style={{
            flex: 1,
            background: C.surface,
            color: C.text,
            border: `1px solid ${C.border}`,
            borderRadius: 8,
            padding: '8px 12px',
            fontSize: 13,
            resize: 'none',
            outline: 'none',
            fontFamily: 'system-ui',
            opacity: !coverage?.complete ? 0.6 : 1,
          }}
        />

        {/* Token count picker */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 3, alignItems: 'center' }}>
          <label style={{ fontSize: 10, color: C.dim }}>max</label>
          <input
            type="number"
            min={1}
            max={256}
            value={maxTokens}
            onChange={e => setMaxTokens(Math.max(1, Math.min(256, Number(e.target.value))))}
            style={{
              width: 44,
              background: C.surface,
              color: C.text,
              border: `1px solid ${C.border}`,
              borderRadius: 6,
              padding: '5px 4px',
              fontSize: 12,
              textAlign: 'center',
            }}
          />
        </div>

        <button
          onClick={send}
          disabled={!canSend}
          style={{
            background: canSend ? C.accent : C.accentDim,
            color: canSend ? '#fff' : C.dim,
            border: 'none',
            borderRadius: 8,
            padding: '10px 18px',
            fontSize: 13,
            cursor: canSend ? 'pointer' : 'default',
            opacity: canSend ? 1 : 0.6,
            alignSelf: 'flex-end',
          }}
        >
          {sending ? '…' : 'Send'}
        </button>
      </div>

      {/* Inline hint when coverage is incomplete */}
      {coverage && !coverage.complete && !coverageLoading && (
        <div style={{ fontSize: 11, color: C.red, marginTop: 6 }}>
          Inference requires all transformer blocks to be covered. Host more blocks or connect peers that cover the missing ranges.
        </div>
      )}
    </div>
  )
}
