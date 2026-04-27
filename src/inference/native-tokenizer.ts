import { AsyncVocabRunner } from './native-worker'

export interface ChatTurn {
  role: 'user' | 'assistant' | 'system'
  content: string
}

/** Thrown when the model has no embedded chat template. Surfaceable to the UI. */
export class ChatTemplateUnavailableError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ChatTemplateUnavailableError'
  }
}

export interface Tokenizer {
  readonly vocabSize: number
  readonly bosTokenId: number
  readonly eosTokenId: number
  readonly endOfTurnTokenId: number | undefined
  /** Encode text to token IDs using byte-level fallback */
  encode(text: string): Promise<Int32Array>
  /** Encode a user chat message with model-specific chat template */
  encodeChat(userMessage: string): Promise<Int32Array>
  /** Encode a multi-turn conversation using the model's chat template. */
  encodeChatMulti(turns: ChatTurn[]): Promise<Int32Array>
  /** Decode a single token ID to its string representation */
  decodeToken(id: number): Promise<string>
  decode(ids: number[]): Promise<string>
}

export class NativeTokenizer implements Tokenizer {
  readonly vocabSize: number
  readonly bosTokenId: number
  readonly eosTokenId: number
  readonly endOfTurnTokenId: number | undefined

  constructor(
    private readonly runner: AsyncVocabRunner,
    specialTokens: { bosId: number; eosId: number; eotId: number; vocabSize: number },
  ) {
    this.vocabSize        = specialTokens.vocabSize
    this.bosTokenId       = specialTokens.bosId
    this.eosTokenId       = specialTokens.eosId
    this.endOfTurnTokenId = specialTokens.eotId >= 0 ? specialTokens.eotId : undefined
  }

  encode(text: string): Promise<Int32Array> {
    return this.runner.tokenize(text, true, false)
  }

  async encodeChat(userMessage: string): Promise<Int32Array> {
    const formatted = await this.runner.applyChatTemplate(userMessage)
    return this.runner.tokenize(formatted, false, true)
  }

  async encodeChatMulti(turns: ChatTurn[]): Promise<Int32Array> {
    let formatted: string
    try {
      formatted = await this.runner.applyChatTemplateMulti(turns)
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      // Sentinel substring is owned by the C++ helper. Keep this string in sync
      // with the throw site in native/src/vocab_context.cpp::vocab_apply_chat_template_multi.
      if (msg.includes('no embedded chat template')) {
        throw new ChatTemplateUnavailableError(msg)
      }
      throw err
    }
    // parseSpecial=true so templated special tokens (<|im_end|>, <|eot_id|>, ...) become real IDs.
    return this.runner.tokenize(formatted, false, true)
  }

  decodeToken(id: number): Promise<string> {
    return this.runner.tokenToPiece(id)
  }

  async decode(ids: number[]): Promise<string> {
    const pieces = await Promise.all(ids.map(id => this.runner.tokenToPiece(id)))
    return pieces.join('')
  }
}

export async function loadNativeTokenizer(filePath: string): Promise<NativeTokenizer> {
  const runner = await AsyncVocabRunner.create(filePath)
  const specialTokens = await runner.getSpecialTokens()
  return new NativeTokenizer(runner, specialTokens)
}

export async function freeNativeTokenizer(t: NativeTokenizer): Promise<void> {
  await (t as any).runner.dispose()
}
