import { ipcMain, BrowserWindow, app } from 'electron'
import type { ApiServer, LogEntry } from './api-server'
import { loadConfig, saveConfig, generateKey } from './api-config'
import { isEnabledInClaude, enableInClaude, disableInClaude } from './claude-settings'

export function setupApiServerIPC(server: ApiServer): void {
  const userDataPath = app.getPath('userData')

  server.onLog((entry: LogEntry) => {
    BrowserWindow.getAllWindows().forEach(w => {
      w.webContents.send('opencoral:api-server-log', entry)
    })
  })

  ipcMain.handle('opencoral:api-server-status', () => ({
    running: server.running,
    port: server.port,
    apiKey: server.apiKey,
    claudeEnabled: isEnabledInClaude(userDataPath),
    endpoints: [
      `POST http://localhost:${server.port}/v1/chat/completions`,
      `POST http://localhost:${server.port}/v1/messages`,
      `GET  http://localhost:${server.port}/v1/models`,
      `GET  http://localhost:${server.port}/health`,
    ],
  }))

  ipcMain.handle('opencoral:api-server-toggle', async (_e, enable: boolean) => {
    const cfg = loadConfig(userDataPath)
    if (enable && !server.running) {
      try {
        await server.start()
        saveConfig(userDataPath, { ...cfg, enabled: true })
      } catch (err) {
        broadcastApiServerStatus(server)
        throw err
      }
    } else if (!enable && server.running) {
      await server.stop()
      saveConfig(userDataPath, { ...cfg, enabled: false })
    }
    broadcastApiServerStatus(server)
  })

  ipcMain.handle('opencoral:api-server-set-port', async (_e, port: number) => {
    if (server.running) throw new Error('Stop the server before changing the port')
    const cfg = loadConfig(userDataPath)
    const updated = { ...cfg, port }
    saveConfig(userDataPath, updated)
    server.updateConfig(updated)
    broadcastApiServerStatus(server)
  })

  ipcMain.handle('opencoral:api-server-regen-key', async () => {
    const cfg = loadConfig(userDataPath)
    const updated = { ...cfg, apiKey: generateKey() }
    saveConfig(userDataPath, updated)
    server.updateConfig(updated)
    broadcastApiServerStatus(server)
  })

  ipcMain.handle('opencoral:api-server-claude-toggle', async () => {
    if (isEnabledInClaude(userDataPath)) {
      disableInClaude(userDataPath)
    } else {
      if (!server.running) throw new Error('Start the server before enabling Claude Code integration')
      enableInClaude(userDataPath, server.port, server.apiKey)
    }
    broadcastApiServerStatus(server)
  })
}

export function broadcastApiServerStatus(server: ApiServer): void {
  const userDataPath = app.getPath('userData')
  const payload = {
    running: server.running,
    port: server.port,
    apiKey: server.apiKey,
    claudeEnabled: isEnabledInClaude(userDataPath),
  }
  BrowserWindow.getAllWindows().forEach(w => {
    w.webContents.send('opencoral:api-server-status-push', payload)
  })
}
