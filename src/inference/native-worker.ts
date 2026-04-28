import { Worker } from 'worker_threads'
import { getNativePath } from './native-loader'

/**
 * Worker script loaded via eval.  Loads the native addon inside a dedicated
 * thread and executes operations posted from the main thread, keeping the
 * Electron main process responsive during heavy compute.
 */
const WORKER_SCRIPT = `
'use strict';
const { parentPort, workerData } = require('worker_threads');
const native = require(workerData.addonPath);

parentPort.on('message', (msg) => {
  try {
    const result = native[msg.op](...msg.args);
    const transfer = [];
    if (result && result.buffer instanceof ArrayBuffer) {
      transfer.push(result.buffer);
    }
    parentPort.postMessage({ id: msg.id, result }, transfer);
  } catch (err) {
    parentPort.postMessage({ id: msg.id, error: err.message });
  }
});
parentPort.postMessage({ ready: true });
`

interface PendingCall {
  resolve: (value: unknown) => void
  reject: (error: Error) => void
}

type AsyncBlockRunnerBase = {
  blockStart: number
  blockEnd: number
  totalBlocks: number
  hiddenSize: number
}

export type AsyncBlockRunnerOptions = AsyncBlockRunnerBase & (
  | { modelPath: string; modelPaths?: never }
  | { modelPaths: string[]; modelPath?: never }
)

/**
 * Drop-in async replacement for BlockRunner that runs all native compute on a
 * worker thread.  Every method returns a Promise so the main thread's event
 * loop is never blocked.
 */
export class AsyncBlockRunner {
  private worker: Worker
  private pending = new Map<number, PendingCall>()
  private nextId = 1
  private _handle = 0
  private _disposed = false
  private _vocabSize = 0

  readonly blockStart: number
  readonly blockEnd: number
  readonly hiddenSize: number

  /* Use AsyncBlockRunner.create() instead. */
  private constructor(blockStart: number, blockEnd: number, hiddenSize: number) {
    this.blockStart = blockStart
    this.blockEnd = blockEnd
    this.hiddenSize = hiddenSize

    const addonPath = getNativePath()
    this.worker = new Worker(WORKER_SCRIPT, {
      eval: true,
      workerData: { addonPath },
    })

    this.worker.on('error', (err) => {
      for (const [, p] of this.pending) p.reject(err)
      this.pending.clear()
    })
  }

  private call(op: string, ...args: unknown[]): Promise<unknown> {
    return new Promise((resolve, reject) => {
      const id = this.nextId++
      this.pending.set(id, { resolve, reject })
      this.worker.postMessage({ id, op, args })
    })
  }

  /**
   * Create an AsyncBlockRunner. The model data is loaded on the worker thread
   * so the main thread stays responsive during model loading as well.
   */
  static async create(opts: AsyncBlockRunnerOptions): Promise<AsyncBlockRunner> {
    const runner = new AsyncBlockRunner(opts.blockStart, opts.blockEnd, opts.hiddenSize)

    // Wait for the worker to finish loading the native addon
    await new Promise<void>((resolve) => {
      const onMessage = (msg: { ready?: boolean }): void => {
        if (msg.ready) {
          runner.worker.off('message', onMessage)
          // Install the normal message handler
          runner.worker.on('message', (m: { id: number; result?: unknown; error?: string }) => {
            const p = runner.pending.get(m.id)
            if (!p) return
            runner.pending.delete(m.id)
            if (m.error) p.reject(new Error(m.error))
            else p.resolve(m.result)
          })
          resolve()
        }
      }
      runner.worker.on('message', onMessage)
    })

    if (opts.modelPaths && opts.modelPaths.length > 1) {
      runner._handle = (await runner.call(
        'loadBlockRangeSharded',
        opts.modelPaths,
        opts.blockStart,
        opts.blockEnd,
        opts.totalBlocks,
      )) as number
    } else {
      const path = opts.modelPath ?? opts.modelPaths?.[0]
      if (!path) throw new Error('AsyncBlockRunner: modelPath or modelPaths required')
      runner._handle = (await runner.call(
        'loadBlockRange',
        path,
        opts.blockStart,
        opts.blockEnd,
        opts.totalBlocks,
      )) as number
    }

    runner._vocabSize = (await runner.call('getVocabSize', runner._handle)) as number

    return runner
  }

  get vocabSize(): number {
    return this._vocabSize
  }

  forward(input: Float32Array, nTokens: number): Promise<Float32Array> {
    return this.call('runForward', this._handle, input, nTokens) as Promise<Float32Array>
  }

  embedTokens(tokenIds: Int32Array): Promise<Float32Array> {
    return this.call('embedTokens', this._handle, tokenIds) as Promise<Float32Array>
  }

  projectToLogits(hidden: Float32Array, nTokens: number): Promise<Float32Array> {
    return this.call('projectToLogits', this._handle, hidden, nTokens) as Promise<Float32Array>
  }

  openSession(maxLength: number): Promise<number> {
    return this.call('openSession', this._handle, maxLength) as Promise<number>
  }

  closeSession(sessionId: number): Promise<void> {
    return this.call('closeSession', this._handle, sessionId) as Promise<void>
  }

  sessionForward(sessionId: number, input: Float32Array, nNewTokens: number): Promise<Float32Array> {
    return this.call('sessionForward', this._handle, sessionId, input, nNewTokens) as Promise<Float32Array>
  }

  sessionDecodeLogits(sessionId: number, tokenIds: Int32Array): Promise<Float32Array> {
    return this.call('sessionDecodeLogits', this._handle, sessionId, tokenIds) as Promise<Float32Array>
  }

  sessionDecodeLogitsAll(sessionId: number, tokenIds: Int32Array): Promise<Float32Array> {
    return this.call('sessionDecodeLogitsAll', this._handle, sessionId, tokenIds) as Promise<Float32Array>
  }

  sessionRollback(sessionId: number, newNPast: number): Promise<void> {
    return this.call('sessionRollback', this._handle, sessionId, newNPast) as Promise<void>
  }

  async dispose(): Promise<void> {
    if (this._disposed) return
    this._disposed = true
    await this.call('freeBlockRange', this._handle)
    await this.worker.terminate()
  }
}

export class AsyncVocabRunner {
  private worker: Worker
  private pending = new Map<number, PendingCall>()
  private nextId = 1
  private _handle = 0
  private _disposed = false

  private constructor() {
    const addonPath = getNativePath()
    this.worker = new Worker(WORKER_SCRIPT, {
      eval: true,
      workerData: { addonPath },
    })
    this.worker.on('error', (err) => {
      for (const [, p] of this.pending) p.reject(err)
      this.pending.clear()
    })
  }

  private call(op: string, ...args: unknown[]): Promise<unknown> {
    return new Promise((resolve, reject) => {
      const id = this.nextId++
      this.pending.set(id, { resolve, reject })
      this.worker.postMessage({ id, op, args })
    })
  }

  static async create(modelPath: string): Promise<AsyncVocabRunner> {
    const runner = new AsyncVocabRunner()
    await new Promise<void>((resolve) => {
      const onMessage = (msg: { ready?: boolean }): void => {
        if (msg.ready) {
          runner.worker.off('message', onMessage)
          runner.worker.on('message', (m: { id: number; result?: unknown; error?: string }) => {
            const p = runner.pending.get(m.id)
            if (!p) return
            runner.pending.delete(m.id)
            if (m.error) p.reject(new Error(m.error))
            else p.resolve(m.result)
          })
          resolve()
        }
      }
      runner.worker.on('message', onMessage)
    })
    runner._handle = (await runner.call('loadVocab', modelPath)) as number
    return runner
  }

  tokenize(text: string, addSpecial: boolean, parseSpecial: boolean): Promise<Int32Array> {
    return this.call('nativeTokenize', this._handle, text, addSpecial, parseSpecial) as Promise<Int32Array>
  }

  tokenToPiece(tokenId: number): Promise<string> {
    return this.call('nativeTokenToPiece', this._handle, tokenId) as Promise<string>
  }

  applyChatTemplate(userMessage: string): Promise<string> {
    return this.call('nativeApplyChatTemplate', this._handle, userMessage) as Promise<string>
  }

  applyChatTemplateMulti(turns: { role: string; content: string }[]): Promise<string> {
    return this.call('nativeApplyChatTemplateMulti', this._handle, turns) as Promise<string>
  }

  getSpecialTokens(): Promise<{ bosId: number; eosId: number; eotId: number; vocabSize: number }> {
    return this.call('nativeGetSpecialTokens', this._handle) as Promise<{ bosId: number; eosId: number; eotId: number; vocabSize: number }>
  }

  async dispose(): Promise<void> {
    if (this._disposed) return
    this._disposed = true
    await this.call('freeVocab', this._handle)
    await this.worker.terminate()
  }
}
