import { AsyncVocabRunner } from './native-worker'

export interface Tokenizer {
  readonly vocabSize: number
  readonly bosTokenId: number
  readonly eosTokenId: number
  readonly endOfTurnTokenId: number | undefined
  /** Encode text to token IDs using byte-level fallback */
  encode(text: string): Promise<Int32Array>
  /** Encode a user chat message with model-specific chat template */
  encodeChat(userMessage: string): Promise<Int32Array>
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
