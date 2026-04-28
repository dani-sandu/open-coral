import type { AsyncBlockRunner } from './native-worker'
import type { Embedder } from './embedder'

export class RunnerEmbedder implements Embedder {
  constructor(private readonly runner: AsyncBlockRunner) {}

  get nEmbd(): number {
    return this.runner.hiddenSize
  }

  embed(tokenIds: Int32Array): Promise<Float32Array> {
    return this.runner.embedTokens(tokenIds)
  }
}
