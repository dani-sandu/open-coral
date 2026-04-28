// src/main/session-store.ts

import { promises as fs } from 'fs'
import { join } from 'path'

export interface InferenceTrace {
  // Mirror of InferenceResult fields kept on persisted assistant messages.
  // Same shape as src/main/block-host.ts InferenceResult — kept duplicated for the same reason.
  prompt: string
  generatedText: string
  generatedTokens: number
  nEmbd: number
  chainSteps: { peerId: string; blockStart: number; blockEnd: number; durationMs: number }[]
  totalDurationMs: number
  specDraftTokens?: number
  specAcceptedTokens?: number
  specAcceptanceRate?: number
}

export interface PersistedChatMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
  result?: InferenceTrace
  error?: string
  timestamp: number
}

export interface PersistedChatSession {
  schemaVersion: 1
  id: string
  title: string
  createdAt: number
  updatedAt: number
  messages: PersistedChatMessage[]
}

export interface SessionSummary {
  id: string
  title: string
  createdAt: number
  updatedAt: number
  messageCount: number
  corrupt?: boolean
}

interface IndexFile {
  schemaVersion: 1
  sessions: SessionSummary[]
}

const INDEX_FILE = 'index.json'

export class SessionStore {
  private readonly baseDir: string
  private index: IndexFile = { schemaVersion: 1, sessions: [] }
  private pending = new Map<string, { session: PersistedChatSession; timer: NodeJS.Timeout }>()
  private readonly debounceMs: number

  constructor(baseDir: string, opts: { debounceMs?: number } = {}) {
    this.baseDir = baseDir
    this.debounceMs = opts.debounceMs ?? 500
  }

  async init(): Promise<void> {
    await fs.mkdir(this.baseDir, { recursive: true })
    try {
      const raw = await fs.readFile(join(this.baseDir, INDEX_FILE), 'utf-8')
      const parsed = JSON.parse(raw) as IndexFile
      if (parsed.schemaVersion !== 1) {
        console.warn(`[SessionStore] index schemaVersion ${parsed.schemaVersion} unsupported — starting fresh`)
        this.index = { schemaVersion: 1, sessions: [] }
        return
      }
      this.index = parsed
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'ENOENT') {
        console.warn(`[SessionStore] index unreadable, starting fresh: ${(err as Error).message}`)
      }
      this.index = { schemaVersion: 1, sessions: [] }
    }
  }

  async list(): Promise<SessionSummary[]> {
    return this.index.sessions.slice()
  }

  async load(id: string): Promise<PersistedChatSession | null> {
    // If a pending write exists, the in-memory copy is the freshest source of truth.
    const pending = this.pending.get(id)
    if (pending) return pending.session

    try {
      const raw = await fs.readFile(this.sessionPath(id), 'utf-8')
      const parsed = JSON.parse(raw) as PersistedChatSession
      if (parsed.schemaVersion !== 1) {
        this.markCorrupt(id, `unsupported schemaVersion ${parsed.schemaVersion}`)
        return null
      }
      return parsed
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') return null
      this.markCorrupt(id, (err as Error).message)
      return null
    }
  }

  async save(session: PersistedChatSession): Promise<void> {
    // Replace any pending write for this id with the newer payload, reset the timer.
    const existing = this.pending.get(session.id)
    if (existing) clearTimeout(existing.timer)

    const timer = setTimeout(() => {
      this.flushOne(session.id).catch(err => {
        console.error(`[SessionStore] background flush failed for ${session.id}: ${(err as Error).message}`)
      })
    }, this.debounceMs)

    this.pending.set(session.id, { session, timer })
  }

  async delete(id: string): Promise<void> {
    const pending = this.pending.get(id)
    if (pending) { clearTimeout(pending.timer); this.pending.delete(id) }

    this.index.sessions = this.index.sessions.filter(s => s.id !== id)
    await this.writeIndex()

    try { await fs.unlink(this.sessionPath(id)) }
    catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err
    }
  }

  /** Force all pending debounced writes to disk synchronously. Call on app.before-quit. */
  async flush(): Promise<void> {
    const ids = Array.from(this.pending.keys())
    // Pass skipIndex=true so each flushOne only writes its session file and
    // updates the in-memory summary; the index file is written ONCE at the end.
    // This avoids a race where parallel flushOne calls collide on index.json.tmp.
    await Promise.all(ids.map(id => this.flushOne(id, true)))
    await this.writeIndex()
  }

  private async flushOne(id: string, skipIndex = false): Promise<void> {
    const entry = this.pending.get(id)
    if (!entry) return
    clearTimeout(entry.timer)
    this.pending.delete(id)

    const path = this.sessionPath(id)
    const tmp = path + '.tmp'
    await fs.writeFile(tmp, JSON.stringify(entry.session, null, 2), 'utf-8')
    // fs.rename is atomic on POSIX and on Windows when target is on the same volume.
    await fs.rename(tmp, path)

    this.upsertSummary({
      id: entry.session.id,
      title: entry.session.title,
      createdAt: entry.session.createdAt,
      updatedAt: entry.session.updatedAt,
      messageCount: entry.session.messages.length,
    })
    if (!skipIndex) await this.writeIndex()
  }

  private upsertSummary(s: SessionSummary): void {
    const i = this.index.sessions.findIndex(x => x.id === s.id)
    if (i === -1) this.index.sessions.unshift(s)
    else this.index.sessions[i] = s
  }

  private async writeIndex(): Promise<void> {
    const tmp = join(this.baseDir, INDEX_FILE + '.tmp')
    await fs.writeFile(tmp, JSON.stringify(this.index, null, 2), 'utf-8')
    await fs.rename(tmp, join(this.baseDir, INDEX_FILE))
  }

  private markCorrupt(id: string, reason: string): void {
    console.warn(`[SessionStore] session ${id} corrupt: ${reason}`)
    const entry = this.index.sessions.find(s => s.id === id)
    if (entry) entry.corrupt = true
    // Best-effort persistence; log failure instead of silently ignoring.
    this.writeIndex().catch(err => {
      console.warn(`[SessionStore] failed to persist corruption marker for ${id}: ${(err as Error).message}`)
    })
  }

  private sessionPath(id: string): string {
    // Defensive: reject ids with path separators or '..' to prevent traversal.
    if (id.includes('/') || id.includes('\\') || id.includes('..')) {
      throw new Error(`SessionStore: invalid session id "${id}"`)
    }
    return join(this.baseDir, `${id}.json`)
  }
}
