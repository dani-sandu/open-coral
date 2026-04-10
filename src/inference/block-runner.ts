import { getNative } from './native-loader'

export interface BlockRunnerOptions {
  /** Absolute path to the GGUF model file */
  modelPath: string
  /** First transformer block index this runner hosts (inclusive) */
  blockStart: number
  /** Last transformer block index this runner hosts (inclusive) */
  blockEnd: number
  /** Total number of transformer blocks in the full model */
  totalBlocks: number
  /** Hidden dimension size (n_embd) — used for input validation */
  hiddenSize: number
}

/**
 * Loads transformer blocks [blockStart..blockEnd] from a GGUF model file
 * and runs hidden-state tensors through them via the native C++ addon.
 *
 * Call dispose() when done to free native memory.
 */
export class BlockRunner {
  private _handle = 0  // 0 is never a valid native handle (addon starts at 1)
  private _disposed = false

  readonly blockStart: number
  readonly blockEnd: number
  readonly hiddenSize: number

  constructor(opts: BlockRunnerOptions) {
    this._handle    = getNative().loadBlockRange(
      opts.modelPath, opts.blockStart, opts.blockEnd, opts.totalBlocks
    )
    this.blockStart = opts.blockStart
    this.blockEnd   = opts.blockEnd
    this.hiddenSize = opts.hiddenSize
  }

  /**
   * Run a forward pass through the loaded blocks.
   *
   * @param input    Float32Array of length `nTokens × hiddenSize`
   * @param nTokens  Number of tokens in the batch
   * @returns        Float32Array of length `nTokens × hiddenSize`
   */
  forward(input: Float32Array, nTokens: number): Float32Array {
    if (this._disposed) {
      throw new Error('BlockRunner has been disposed')
    }
    const expected = nTokens * this.hiddenSize
    if (input.length !== expected) {
      throw new Error(
        `Input length ${input.length} does not match expected ${expected} ` +
        `(${nTokens} tokens × ${this.hiddenSize} hidden size)`
      )
    }
    return getNative().runForward(this._handle, input, nTokens)
  }

  /** Release native resources. Safe to call multiple times. */
  dispose(): void {
    if (!this._disposed) {
      getNative().freeBlockRange(this._handle)
      this._disposed = true
    }
  }
}
