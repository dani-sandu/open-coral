import {
  SpeculativeSession,
  LocalVerificationBackend,
  DEFAULT_SPEC_CONFIG,
  type GenerateResult,
} from '../inference/speculative-session'
import type { AsyncBlockRunner } from '../inference/native-worker'
import type { NativeTokenizer } from '../inference/native-tokenizer'

export interface InferenceOptions {
  runner: AsyncBlockRunner
  tokenizer: NativeTokenizer
  prompt: string
  maxTokens: number
  requestId: string
}

export interface InferenceOutput {
  genResult: GenerateResult
  promptLength: number
  inferenceDurationMs: number
}

export async function runInference(opts: InferenceOptions): Promise<InferenceOutput> {
  const { runner, tokenizer, prompt, maxTokens, requestId } = opts
  const promptIds = await tokenizer.encodeChat(prompt)
  console.log(`[OpenCoral] [${requestId}] Prompt tokens (${promptIds.length})`)

  const sessionId = await runner.openSession(promptIds.length + maxTokens)
  try {
    const backend = new LocalVerificationBackend(runner, sessionId)
    const session = new SpeculativeSession(
      backend,
      tokenizer.eosTokenId,
      tokenizer.endOfTurnTokenId,
      DEFAULT_SPEC_CONFIG,
    )

    const t1 = Date.now()
    const genResult = await session.generate(promptIds, maxTokens)
    const inferenceDurationMs = Date.now() - t1

    if (genResult.specDraftTokens > 0) {
      const rate = (genResult.specAcceptedTokens / genResult.specDraftTokens * 100).toFixed(1)
      console.log(`[OpenCoral] [${requestId}] Spec decoding: ${genResult.specAcceptedTokens}/${genResult.specDraftTokens} accepted (${rate}%)`)
    }

    return { genResult, promptLength: promptIds.length, inferenceDurationMs }
  } finally {
    await runner.closeSession(sessionId)
  }
}

export interface InferenceStreamOptions {
  runner: AsyncBlockRunner
  tokenizer: NativeTokenizer
  promptIds: Int32Array
  maxTokens: number
  requestId: string
}

export async function* runInferenceStream(opts: InferenceStreamOptions): AsyncGenerator<string> {
  const { runner, tokenizer, promptIds, maxTokens, requestId } = opts
  console.log(`[OpenCoral] [${requestId}] Stream inference, prompt tokens: ${promptIds.length}`)

  const sessionId = await runner.openSession(promptIds.length + maxTokens)
  try {
    const backend = new LocalVerificationBackend(runner, sessionId)
    const session = new SpeculativeSession(
      backend,
      tokenizer.eosTokenId,
      tokenizer.endOfTurnTokenId,
      DEFAULT_SPEC_CONFIG,
    )
    for await (const tokenId of session.generateTokens(promptIds, maxTokens)) {
      yield await tokenizer.decodeToken(tokenId)
    }
  } finally {
    await runner.closeSession(sessionId)
  }
}
