import { app, BrowserWindow, shell, ipcMain } from 'electron'
import { join } from 'path'
import { createCoralNode, type CoralNode } from '../p2p/node'
import { inspectNetwork, type NetworkState } from '../p2p/network-inspector'
import { setupModelIPC, getCurrentModel } from './model-manager'
import { setupBlockHostIPC, setupInferenceIPC, setupCoverageIPC, getActiveHost } from './block-host'
import { setupHuggingFaceIPC } from './huggingface'
import { registerModelInfoHandler, queryPeerModelInfo } from '../p2p/model-announce'
import { DiscoveredModels } from '../p2p/discovered-models'
import type { NetworkModelEntry } from '../p2p/discovered-models'

let coralNode: CoralNode | null = null
let localBlocks: { start: number; end: number }[] = []
const discoveredModels = new DiscoveredModels()

async function startCoralNode(): Promise<void> {
  try {
    coralNode = await createCoralNode()
    const node = coralNode  // capture for use in closures
    console.log(`[Coral] P2P node started: ${node.peerId}`)
    console.log(`[Coral] Listening on: ${node.multiaddrs.join(', ')}`)

    // Register model-info protocol
    await registerModelInfoHandler(node.libp2p, () => {
      const host = getActiveHost()
      const model = getCurrentModel()
      if (!host || !model?.repoId || !model?.hfFilename) return null
      return {
        repoId:       model.repoId,
        hfFilename:   model.hfFilename,
        blockStart:   host.state.blockStart,
        blockEnd:     host.state.blockEnd,
        totalBlocks:  model.totalBlocks,
        hiddenSize:   model.hiddenSize,
        architecture: model.architecture,
      }
    })

    // Auto-query newly connected peers for their model info
    node.libp2p.addEventListener('peer:connect', async (evt) => {
      const peerId = evt.detail
      const info = await queryPeerModelInfo(node.libp2p, peerId)
      if (info) {
        discoveredModels.update(peerId.toString(), info)
      }
    })

    // Remove disconnected peers from the registry
    node.libp2p.addEventListener('peer:disconnect', (evt) => {
      const peerId = evt.detail
      discoveredModels.remove(peerId.toString())
    })
  } catch (err) {
    console.error('[Coral] Failed to start P2P node:', err)
  }
}

function setupNetworkModelIPC(): void {
  ipcMain.handle('coral:discover-network-models', async (): Promise<NetworkModelEntry[]> => {
    const node = coralNode
    if (!node) return []

    // Query all currently connected peers — updates the persistent registry
    const connectedPeers = node.libp2p.getPeers()
    await Promise.all(
      connectedPeers.map(async (peerId) => {
        const info = await queryPeerModelInfo(node.libp2p, peerId)
        if (info) {
          discoveredModels.update(peerId.toString(), info)
        } else {
          discoveredModels.remove(peerId.toString())
        }
      })
    )

    // Always inject the local model into the registry if hosting (so aggregate() computes complete correctly)
    const localModel = getCurrentModel()
    const host = getActiveHost()
    if (localModel?.repoId && localModel?.hfFilename && host) {
      discoveredModels.update('local', {
        repoId:       localModel.repoId,
        hfFilename:   localModel.hfFilename,
        blockStart:   host.state.blockStart,
        blockEnd:     host.state.blockEnd,
        totalBlocks:  localModel.totalBlocks,
        hiddenSize:   localModel.hiddenSize,
        architecture: localModel.architecture,
      })
    } else {
      // Not hosting — remove any stale local entry
      discoveredModels.remove('local')
    }

    return discoveredModels.aggregate()
  })
}

function setupIPC(): void {
  setupModelIPC()
  setupBlockHostIPC(() => coralNode, (blocks) => { localBlocks = blocks })
  setupInferenceIPC(() => coralNode)
  setupCoverageIPC(() => coralNode)
  setupHuggingFaceIPC()
  setupNetworkModelIPC()

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
