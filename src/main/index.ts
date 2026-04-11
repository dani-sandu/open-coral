import { app, BrowserWindow, shell, ipcMain } from 'electron'
import { join } from 'path'
import { createCoralNode, type CoralNode } from '../p2p/node'
import { inspectNetwork, type NetworkState } from '../p2p/network-inspector'
import { setupModelIPC } from './model-manager'
import { setupBlockHostIPC, setupInferenceDemoIPC } from './block-host'
import { setupHuggingFaceIPC } from './huggingface'

let coralNode: CoralNode | null = null
let localBlocks: { start: number; end: number }[] = []

async function startCoralNode(): Promise<void> {
  try {
    coralNode = await createCoralNode()
    console.log(`[Coral] P2P node started: ${coralNode.peerId}`)
    console.log(`[Coral] Listening on: ${coralNode.multiaddrs.join(', ')}`)
  } catch (err) {
    console.error('[Coral] Failed to start P2P node:', err)
  }
}

function setupIPC(): void {
  setupModelIPC()
  setupBlockHostIPC(() => coralNode, (blocks) => { localBlocks = blocks })
  setupInferenceDemoIPC(() => coralNode)
  setupHuggingFaceIPC()

  ipcMain.handle('coral:get-network-state', (): NetworkState | null => {
    if (!coralNode) return null
    return inspectNetwork(coralNode, localBlocks)
  })
}

function createWindow(): void {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    title: 'Coral',
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false
    }
  })

  win.on('ready-to-show', () => win.show())

  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })

  if (process.env['ELECTRON_RENDERER_URL']) {
    win.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    win.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

app.whenReady().then(async () => {
  setupIPC()
  await startCoralNode()
  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', async () => {
  if (coralNode) {
    await coralNode.stop()
    coralNode = null
  }
})
