import { app, BrowserWindow, shell, ipcMain, Menu } from 'electron'
import { join } from 'path'
import { createOpenCoralNode, getBootstrapPeers, type OpenCoralNode } from '../p2p/node'
import { inspectNetwork, type NetworkState } from '../p2p/network-inspector'
import { setupModelIPC, getCurrentModel, subscribeModelChange } from './model-manager'
import { setupBlockHostIPC, setupInferenceIPC, setupCoverageIPC, getActiveHost, getShimRunner, getCachedTokenizer } from './block-host'
import { setupHuggingFaceIPC } from './huggingface'
import { registerModelInfoHandler, queryPeerModelInfo } from '../p2p/model-announce'
import { DiscoveredModels } from '../p2p/discovered-models'
import type { NetworkModelEntry } from '../p2p/discovered-models'
import { PeerLatencyTracker } from '../p2p/peer-latency'
import { loadOrCreateIdentity } from './identity'
import type { NodeIdentity } from './identity'
import { SessionStore } from './session-store'
import {
  ChatSessionManager,
  SequenceManagerChainPlanner,
} from './chat-session-manager'
import {
  setupChatSessionIPC,
  broadcastSessionPhase,
  broadcastSessionInvalidated,
} from './chat-session-ipc'
import { SequenceManager } from '../inference/sequence-manager'

let openCoralNode: OpenCoralNode | null = null
let localBlocks: { start: number; end: number }[] = []
const discoveredModels = new DiscoveredModels()
const latencyTracker = new PeerLatencyTracker()
let nodeIdentity: NodeIdentity | null = null
let sessionStore: SessionStore | null = null
let chatSessionManager: ChatSessionManager | null = null

export function getPeerBlockRange(peerId: string): { blockStart: number; blockEnd: number } | null {
  return discoveredModels.getPeerRange(peerId)
}

export function getLatencyTracker(): PeerLatencyTracker { return latencyTracker }

export function getNodeIdentity(): NodeIdentity {
  if (!nodeIdentity) throw new Error('Identity not yet loaded')
  return nodeIdentity
}

function buildPlanner(): SequenceManagerChainPlanner | null {
  if (!openCoralNode) return null
  const model = getCurrentModel()
  if (!model) return null
  const sm = new SequenceManager({
    node: openCoralNode,
    localRunner: getActiveHost()?.runner ?? null,
    totalBlocks: model.totalBlocks,
    hiddenSize: model.hiddenSize,
    getPeerBlockRange,
    latencyTracker: getLatencyTracker(),
    identity: getNodeIdentity(),
    repoId: model.repoId,
  })
  return new SequenceManagerChainPlanner({ node: openCoralNode, manager: sm })
}

async function startOpenCoralNode(): Promise<void> {
  try {
    openCoralNode = await createOpenCoralNode()
    const node = openCoralNode  // capture for use in closures
    console.log(`[OpenCoral] P2P node started: ${node.peerId}`)
    console.log(`[OpenCoral] Listening on: ${node.multiaddrs.join(', ')}`)

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
      latencyTracker.forget(peerId.toString())
      chatSessionManager?.notifyPeerDisconnected(peerId.toString())
    })
  } catch (err) {
    console.error('[OpenCoral] Failed to start P2P node:', err)
  }
}

function setupNetworkModelIPC(): void {
  ipcMain.handle('opencoral:discover-network-models', async (): Promise<NetworkModelEntry[]> => {
    const node = openCoralNode
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
  setupBlockHostIPC(() => openCoralNode, (blocks) => { localBlocks = blocks })
  setupInferenceIPC(() => openCoralNode)
  setupCoverageIPC(() => openCoralNode)
  setupHuggingFaceIPC()
  setupNetworkModelIPC()

  // ── Chat session persistence ─────────────────────────────────────────────────
  const sessionsDir = join(app.getPath('userData'), 'sessions')
  sessionStore = new SessionStore(sessionsDir)
  sessionStore.init().catch(err => console.error('[OpenCoral] SessionStore init failed:', err))

  chatSessionManager = new ChatSessionManager({
    store: sessionStore,
    // The shim runner is the one valid for local KV ops (sessionDecodeLogitsAll
    // requires block_end == -1). activeHost.runner is for serving peer inference
    // via the v3 protocol — not local inference. Mirrors the embedRunner
    // precedence in src/main/block-host.ts.
    getRunner: () => getShimRunner() ?? getActiveHost()?.runner ?? null,
    getTokenizer: () => getCachedTokenizer(),
    getPlanner: () => buildPlanner(),
    emitPhase: broadcastSessionPhase,
    emitInvalidation: broadcastSessionInvalidated,
  })

  setupChatSessionIPC(chatSessionManager, sessionStore)

  subscribeModelChange(() => {
    chatSessionManager?.invalidateActive('model-change').catch(() => {})
  })

  ipcMain.handle('opencoral:get-network-state', (): NetworkState | null => {
    if (!openCoralNode) return null

    // Ensure local model info is up to date in discoveredModels
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
      discoveredModels.remove('local')
    }

    // Build a peer→modelInfo map from discoveredModels
    const peerModelMap = new Map<string, { repoId: string; hfFilename: string; blockStart: number; blockEnd: number; totalBlocks: number; architecture: string }>()
    for (const { peerId, info } of discoveredModels.list()) {
      peerModelMap.set(peerId === 'local' ? openCoralNode.peerId : peerId, {
        repoId: info.repoId,
        hfFilename: info.hfFilename,
        blockStart: info.blockStart,
        blockEnd: info.blockEnd,
        totalBlocks: info.totalBlocks,
        architecture: info.architecture,
      })
    }

    // Extract bootstrap peer IDs from multiaddrs so the UI can tag them
    const bootstrapPeerIds = new Set<string>()
    for (const addr of getBootstrapPeers()) {
      const match = addr.match(/\/p2p\/([^/]+)$/)
      if (match) bootstrapPeerIds.add(match[1])
    }

    return inspectNetwork(openCoralNode, localBlocks, peerModelMap, bootstrapPeerIds)
  })
}

function createWindow(): void {
  const iconPath = app.isPackaged
    ? join(process.resourcesPath, 'icon.png')
    : join(__dirname, '../../build/icon.png')

  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    title: 'OpenCoral',
    icon: iconPath,
    backgroundColor: '#1e1e2e',
    autoHideMenuBar: true,
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
  nodeIdentity = await loadOrCreateIdentity(app.getPath('userData'))
  setupIPC()
  await startOpenCoralNode()
  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', async () => {
  if (sessionStore) {
    try { await sessionStore.flush() } catch (err) { console.error('[OpenCoral] sessionStore.flush failed:', err) }
  }
  if (chatSessionManager) {
    try { await chatSessionManager.shutdown() } catch (err) { console.error('[OpenCoral] chatSessionManager.shutdown failed:', err) }
  }
  if (openCoralNode) {
    await openCoralNode.stop()
    openCoralNode = null
  }
})
