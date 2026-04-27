// src/main/chat-session-ipc.ts
// IPC wiring + main→renderer broadcasts for chat sessions. Kept in a separate
// file from chat-session-manager.ts so importing the manager core from tests
// (which run under bun, not electron) does not pull in the electron module.

import { ipcMain, BrowserWindow } from 'electron'
import type { SessionStore, PersistedChatSession, SessionSummary } from './session-store'
import type {
  ChatSessionManager,
  SessionPhaseEvent,
  InvalidationEvent,
  TurnResult,
} from './chat-session-manager'

export function setupChatSessionIPC(mgr: ChatSessionManager, store: SessionStore): void {
  ipcMain.handle('opencoral:list-sessions', async () => {
    return store.list()
  })

  ipcMain.handle('opencoral:get-session', async (_e, id: string) => {
    return store.load(id)
  })

  ipcMain.handle('opencoral:create-session', async (): Promise<SessionSummary> => {
    const id = `session-${Date.now()}`
    const now = Date.now()
    const session: PersistedChatSession = {
      schemaVersion: 1, id, title: 'New chat',
      createdAt: now, updatedAt: now, messages: [],
    }
    await store.save(session)
    await store.flush()
    const summary: SessionSummary = {
      id, title: 'New chat', createdAt: now, updatedAt: now, messageCount: 0,
    }
    broadcastSessionUpdated(summary)
    return summary
  })

  ipcMain.handle('opencoral:send-turn', async (
    _e, sessionId: string, userText: string, maxTokens: number,
  ): Promise<TurnResult> => {
    if (typeof sessionId !== 'string' || sessionId.length === 0) throw new Error('sessionId required')
    if (typeof userText !== 'string' || userText.length === 0) throw new Error('userText required')
    if (!Number.isInteger(maxTokens) || maxTokens <= 0 || maxTokens > 2048) {
      throw new Error(`maxTokens must be a positive integer <= 2048, got ${maxTokens}`)
    }
    try {
      const result = await mgr.openTurn(sessionId, userText, maxTokens)
      const session = await store.load(sessionId)
      if (session) {
        broadcastSessionUpdated({
          id: session.id, title: session.title,
          createdAt: session.createdAt, updatedAt: session.updatedAt,
          messageCount: session.messages.length,
        })
      }
      return result
    } catch (err) {
      const session = await store.load(sessionId)
      if (session) {
        broadcastSessionUpdated({
          id: session.id, title: session.title,
          createdAt: session.createdAt, updatedAt: session.updatedAt,
          messageCount: session.messages.length,
        })
      }
      throw err
    }
  })

  ipcMain.handle('opencoral:delete-session', async (_e, id: string): Promise<void> => {
    if (mgr.activeSessionId() === id) {
      // Tear down silently — broadcastSessionDeleted below already informs the
      // renderer; an extra "context lost" toast would be confusing.
      await mgr.tearDownActive()
    }
    await store.delete(id)
    broadcastSessionDeleted(id)
  })

  ipcMain.handle('opencoral:rename-session', async (_e, id: string, title: string): Promise<void> => {
    const s = await store.load(id)
    if (!s) throw new Error(`Session ${id} not found`)
    s.title = title
    s.updatedAt = Date.now()
    await store.save(s)
    broadcastSessionUpdated({
      id: s.id, title: s.title, createdAt: s.createdAt, updatedAt: s.updatedAt,
      messageCount: s.messages.length,
    })
  })
}

function broadcastSessionUpdated(summary: SessionSummary): void {
  for (const win of BrowserWindow.getAllWindows()) {
    win.webContents.send('opencoral:session-updated', summary)
  }
}

function broadcastSessionDeleted(id: string): void {
  for (const win of BrowserWindow.getAllWindows()) {
    win.webContents.send('opencoral:session-deleted', id)
  }
}

export function broadcastSessionPhase(e: SessionPhaseEvent): void {
  for (const win of BrowserWindow.getAllWindows()) {
    win.webContents.send('opencoral:session-phase', e)
  }
}

export function broadcastSessionInvalidated(e: InvalidationEvent): void {
  for (const win of BrowserWindow.getAllWindows()) {
    win.webContents.send('opencoral:session-invalidated', e)
  }
}
