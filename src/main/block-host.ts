import { ipcMain } from 'electron'
import { BlockRunner } from '../inference/block-runner'
import { BlockRegistry } from '../p2p/block-registry'
import { registerInferenceHandler, sendInferenceRequest } from '../p2p/inference-protocol'
import { getCurrentModel } from './model-manager'
import { SequenceManager } from '../inference/sequence-manager'
import { peerIdFromString } from '@libp2p/peer-id'
import { multiaddr } from '@multiformats/multiaddr'
import type { CoralNode } from '../p2p/node'

export interface HostingState {
  modelPath: string
  blockStart: number
  blockEnd: number
  totalBlocks: number
  hiddenSize: number
}

export interface InferenceDemoResult {
  nTokens: number
  nEmbd: number
  chainSteps: {
    peerId: string
    blockStart: number
    blockEnd: number
    durationMs: number
  }[]
  totalDurationMs: number
  outputNorm: number
}

interface ActiveHost {
  runner: BlockRunner
  registry: BlockRegistry
  state: HostingState
}

let activeHost: ActiveHost | null = null

export function setupBlockHostIPC(
  getNode: () => CoralNode | null,
  onBlocksChanged: (blocks: { start: number; end: number }[]) => void,
): void {
  ipcMain.handle('coral:start-hosting', async (
    _event,
    blockStart: number,
    blockEnd: number,
  ): Promise<void> => {
    const model = getCurrentModel()
    if (!model) throw new Error('No model loaded — call coral:select-model first')

    // Validate tensor name compatibility — the native addon expects llama-style separate Q/K/V projections
    const REQUIRED_SUFFIXES = ['attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight', 'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight']
    const missing = REQUIRED_SUFFIXES.filter(s => !model.blockTensorSuffixes.includes(s))
    if (missing.length > 0) {
      const has = model.blockTensorSuffixes.join(', ')
      throw new Error(
        `This model uses an unsupported architecture (${model.architecture}). ` +
        `The native runner expects separate Q/K/V projections (llama-style) but this model has: ${has}. ` +
        `Missing: ${missing.join(', ')}`
      )
    }

    const node = getNode()
    if (!node) throw new Error('P2P node not running')

    // Stop existing host if any
    if (activeHost) {
      activeHost.runner.dispose()
      activeHost.registry.dispose()
      activeHost = null
    }

    const runner = new BlockRunner({
      modelPath: model.path,
      blockStart,
      blockEnd,
      totalBlocks: model.totalBlocks,
      hiddenSize: model.hiddenSize,
    })

    const registry = new BlockRegistry(node.libp2p, {
      blockStart,
      blockEnd,
    })
    await registry.start()

    // Wire BlockRunner into the inference protocol so peers can use our blocks
    await registerInferenceHandler(node.libp2p, async (input, nTokens) => {
      return runner.forward(input, nTokens)
    })

    activeHost = {
      runner,
      registry,
      state: {
        modelPath: model.path,
        blockStart,
        blockEnd,
        totalBlocks: model.totalBlocks,
        hiddenSize: model.hiddenSize,
      },
    }

    onBlocksChanged([{ start: blockStart, end: blockEnd }])
    console.log(`[Coral] Now hosting blocks ${blockStart}–${blockEnd} from ${model.path}`)
  })

  ipcMain.handle('coral:stop-hosting', async (): Promise<void> => {
    if (!activeHost) return
    activeHost.runner.dispose()
    activeHost.registry.dispose()
    activeHost = null
    onBlocksChanged([])
    console.log('[Coral] Stopped hosting blocks')
  })

  ipcMain.handle('coral:get-hosting-state', (): HostingState | null => {
    return activeHost?.state ?? null
  })
}

export function setupInferenceDemoIPC(getNode: () => CoralNode | null): void {
  ipcMain.handle('coral:run-inference-demo', async (
    _event,
    nTokens: number = 1,
  ): Promise<InferenceDemoResult> => {
    const host = activeHost
    if (!host) throw new Error('Not hosting any blocks — start hosting first')

    const node = getNode()
    if (!node) throw new Error('P2P node not running')

    const mgr = new SequenceManager({
      node,
      localRunner: host.runner,
      totalBlocks: host.state.totalBlocks,
      hiddenSize: host.state.hiddenSize,
    })

    const t0 = Date.now()
    const chain = await mgr.planChain()

    const nEmbd = host.state.hiddenSize
    const input = new Float32Array(nTokens * nEmbd)
    for (let i = 0; i < input.length; i++) input[i] = (Math.random() - 0.5) * 0.1

    const stepResults: InferenceDemoResult['chainSteps'] = []
    let current = input

    for (const step of chain) {
      const stepT0 = Date.now()
      if (step.peerId === 'local') {
        current = host.runner.forward(current, nTokens)
      } else {
        const peerId = peerIdFromString(step.peerId)
        const isConnected = node.libp2p.getPeers().some(p => p.equals(peerId))
        if (!isConnected && step.multiaddr) {
          await node.libp2p.dial(multiaddr(step.multiaddr))
        }
        current = await sendInferenceRequest(node.libp2p, peerId, current, nTokens, nEmbd)
      }
      stepResults.push({
        peerId: step.peerId,
        blockStart: step.blockStart,
        blockEnd: step.blockEnd,
        durationMs: Date.now() - stepT0,
      })
    }

    // L2 norm of output (sanity check value)
    let norm = 0
    for (const v of current) norm += v * v
    norm = Math.sqrt(norm)

    return {
      nTokens,
      nEmbd,
      chainSteps: stepResults,
      totalDurationMs: Date.now() - t0,
      outputNorm: norm,
    }
  })
}

export function getActiveHost(): ActiveHost | null {
  return activeHost
}
