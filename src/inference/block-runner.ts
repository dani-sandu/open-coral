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

  /**
   * Look up token embeddings from the embedding weight matrix.
   * Only available when blockStart === 0 (embedding tensor loaded).
   */
  embedTokens(tokenIds: Int32Array): Float32Array {
    if (this._disposed) throw new Error('BlockRunner has been disposed')
    return getNative().embedTokens(this._handle, tokenIds)
  }

  /**
   * Apply final RMS norm and project hidden states to vocabulary logits.
   * Only available when blockEnd === last block (output tensors loaded).
   *
   * @returns Float32Array of length nTokens × vocabSize
   */
  projectToLogits(hidden: Float32Array, nTokens: number): Float32Array {
    if (this._disposed) throw new Error('BlockRunner has been disposed')
    const expected = nTokens * this.hiddenSize
    if (hidden.length !== expected) {
      throw new Error(
        `Input length ${hidden.length} does not match expected ${expected} ` +
        `(${nTokens} tokens × ${this.hiddenSize} hidden size)`
      )
    }
    return getNative().projectToLogits(this._handle, hidden, nTokens)
  }

  /** Vocabulary size from the native model context. */
  get vocabSize(): number {
    if (this._disposed) throw new Error('BlockRunner has been disposed')
    return getNative().getVocabSize(this._handle)
  }

  /** Allocate a KV cache session for up to maxLength tokens. */
  openSession(maxLength: number): number {
    if (this._disposed) throw new Error('BlockRunner has been disposed')
    return getNative().openSession(this._handle, maxLength)
  }

  /** Free a KV cache session. */
  closeSession(sessionId: number): void {
    if (this._disposed) return
    getNative().closeSession(this._handle, sessionId)
  }

  /** Forward pass with KV caching — only processes nNewTokens new tokens. */
  sessionForward(sessionId: number, input: Float32Array, nNewTokens: number): Float32Array {
    if (this._disposed) throw new Error('BlockRunner has been disposed')
    const expected = nNewTokens * this.hiddenSize
    if (input.length !== expected) {
      throw new Error(
        `Input length ${input.length} does not match expected ${expected} ` +
        `(${nNewTokens} tokens × ${this.hiddenSize} hidden size)`
      )
    }
    return getNative().sessionForward(this._handle, sessionId, input, nNewTokens)
  }

  /**
   * KV-cached full-model decode: token IDs → logits for the last token.
   * Only valid on shim contexts (blockEnd === -1) that carry the full model.
   * Accumulates KV state across calls within the same session.
   *
   * @returns Float32Array of length vocabSize
   */
  sessionDecodeLogits(sessionId: number, tokenIds: Int32Array): Float32Array {
    if (this._disposed) throw new Error('BlockRunner has been disposed')
    return getNative().sessionDecodeLogits(this._handle, sessionId, tokenIds)
  }

  /** Release native resources. Safe to call multiple times. */
  dispose(): void {
    if (!this._disposed) {
      getNative().freeBlockRange(this._handle)
      this._disposed = true
    }
  }
}
