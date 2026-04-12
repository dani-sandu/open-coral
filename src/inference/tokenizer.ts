import type { GGUFHeader } from './types'

export interface Tokenizer {
  readonly vocabSize: number
  readonly bosTokenId: number
  readonly eosTokenId: number
  readonly endOfTurnTokenId: number | undefined
  /** Encode text to token IDs using byte-level fallback */
  encode(text: string): Int32Array
  /** Encode a user chat message with model-specific chat template */
  encodeChat(userMessage: string): Int32Array
  /** Decode a single token ID to its string representation */
  decodeToken(id: number): string
  /** Decode a sequence of token IDs to text */
  decode(ids: number[]): string
}

/**
 * Build the GPT-2 byte-level BPE unicode→byte reverse mapping.
 * GPT-2 BPE maps each byte to a printable Unicode character so all tokens are
 * visible strings. E.g. byte 0x20 (space) → U+0120 (Ġ), byte 0x0A (newline) → U+010A (Ċ).
 * Printable ASCII (33–126) and Latin-1 supplement (161–172, 174–255) map to themselves.
 */
function buildGpt2UnicodeToByteMap(): Map<number, number> {
  const bs: number[] = []
  for (let i = 33; i <= 126; i++) bs.push(i)
  for (let i = 161; i <= 172; i++) bs.push(i)
  for (let i = 174; i <= 255; i++) bs.push(i)
  const cs = [...bs]
  let n = 0
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b)
      cs.push(256 + n)
      n++
    }
  }
  // Reverse: unicode codepoint → byte value
  return new Map(bs.map((b, i) => [cs[i], b]))
}

export function createTokenizer(header: GGUFHeader): Tokenizer {
  const metaGet = (key: string): unknown =>
    header.metadata.find(m => m.key === key)?.value ?? null

  const tokens = metaGet('tokenizer.ggml.tokens') as string[] | null
  if (!tokens || !Array.isArray(tokens)) {
    throw new Error('GGUF file does not contain tokenizer.ggml.tokens')
  }
  // Narrowed binding for closure capture — TypeScript doesn't propagate the null-check
  // into inner functions that close over `tokens` because it sees the original `string[] | null` type.
  const vocab: string[] = tokens

  const tokenizerModel = metaGet('tokenizer.ggml.model') as string | null
  const isGpt2Bpe = tokenizerModel === 'gpt2'
  const unicodeToByte = isGpt2Bpe ? buildGpt2UnicodeToByteMap() : null
  const byteToUnicode = isGpt2Bpe ? new Map([...unicodeToByte!].map(([u, b]) => [b, u])) : null

  const bosTokenId = Number(metaGet('tokenizer.ggml.bos_token_id') ?? 1)
  const eosTokenId = Number(metaGet('tokenizer.ggml.eos_token_id') ?? 2)

  // Build reverse lookup: token string → token ID
  // Also build byte-token map for <0xHH> style tokens
  const tokenToId = new Map<string, number>()
  const byteTokenId = new Map<number, number>()

  for (let i = 0; i < vocab.length; i++) {
    const t = vocab[i]
    tokenToId.set(t, i)
    // Detect byte tokens like <0x00> .. <0xFF>
    const byteMatch = t.match(/^<0x([0-9A-Fa-f]{2})>$/)
    if (byteMatch) {
      byteTokenId.set(parseInt(byteMatch[1], 16), i)
    }
  }

  // Find max token length for greedy matching
  let maxTokenLen = 0
  for (const t of vocab) {
    if (t.length > maxTokenLen) maxTokenLen = t.length
  }

  /**
   * Core greedy longest-match tokenizer. The input text should already have
   * spaces replaced with ▁ where appropriate.
   */
  function tokenizeNormalized(text: string): number[] {
    const ids: number[] = []
    let pos = 0
    while (pos < text.length) {
      let matched = false
      const remaining = text.length - pos
      const tryLen = Math.min(maxTokenLen, remaining)

      for (let len = tryLen; len >= 1; len--) {
        const sub = text.substring(pos, pos + len)
        const id = tokenToId.get(sub)
        if (id !== undefined) {
          ids.push(id)
          pos += len
          matched = true
          break
        }
      }

      if (!matched) {
        const ch = text.codePointAt(pos)!
        const bytes = new TextEncoder().encode(String.fromCodePoint(ch))
        for (const b of bytes) {
          const byteId = byteTokenId.get(b)
          if (byteId !== undefined) {
            ids.push(byteId)
          }
        }
        pos += String.fromCodePoint(ch).length
      }
    }
    return ids
  }

  /** Convert text to the model's token-level representation before greedy matching. */
  function normalizeForVocab(text: string): string {
    if (byteToUnicode) {
      // GPT-2 BPE: encode text as UTF-8 bytes, then map each byte to its GPT-2 unicode char
      const bytes = new TextEncoder().encode(text)
      return Array.from(bytes, b => String.fromCodePoint(byteToUnicode.get(b)!)).join('')
    }
    // SentencePiece: replace spaces with ▁
    return text.replace(/ /g, '\u2581')
  }

  function tokenizeSegment(text: string): number[] {
    return tokenizeNormalized(normalizeForVocab(text))
  }

  function encode(text: string): Int32Array {
    if (isGpt2Bpe) {
      // GPT-2 BPE: no leading space convention, just encode the text directly
      const ids: number[] = [bosTokenId, ...tokenizeSegment(text)]
      return new Int32Array(ids)
    }
    // SentencePiece convention: prepend a space so the first word gets the ▁ prefix
    const ids: number[] = [bosTokenId, ...tokenizeSegment(' ' + text)]
    return new Int32Array(ids)
  }

  /** Encode with model-specific chat template. Supports:
   *  - Gemma: <bos><start_of_turn>user\n{msg}<end_of_turn>\n<start_of_turn>model\n
   *  - LLaMA 3: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
   */
  function encodeChat(userMessage: string): Int32Array {
    // Try LLaMA 3 chat template
    const startHeaderId = tokenToId.get('<|start_header_id|>')
    const endHeaderId = tokenToId.get('<|end_header_id|>')
    const eotId = tokenToId.get('<|eot_id|>')
    if (startHeaderId !== undefined && endHeaderId !== undefined && eotId !== undefined) {
      const ids: number[] = [bosTokenId]
      ids.push(startHeaderId)
      ids.push(...tokenizeSegment('user'))
      ids.push(endHeaderId)
      ids.push(...tokenizeSegment('\n\n' + userMessage))
      ids.push(eotId)
      ids.push(startHeaderId)
      ids.push(...tokenizeSegment('assistant'))
      ids.push(endHeaderId)
      ids.push(...tokenizeSegment('\n\n'))
      return new Int32Array(ids)
    }

    // Try Gemma chat template
    const startOfTurnId = tokenToId.get('<start_of_turn>')
    const endOfTurnId = tokenToId.get('<end_of_turn>')
    if (startOfTurnId !== undefined && endOfTurnId !== undefined) {
      const ids: number[] = [bosTokenId]
      ids.push(startOfTurnId)
      ids.push(...tokenizeSegment('user\n' + userMessage))
      ids.push(endOfTurnId)
      ids.push(...tokenizeSegment('\n'))
      ids.push(startOfTurnId)
      ids.push(...tokenizeSegment('model\n'))
      return new Int32Array(ids)
    }

    // Fallback: plain encode
    return encode(userMessage)
  }

  function decodeToken(id: number): string {
    if (id < 0 || id >= vocab.length) return ''
    const t = vocab[id]
    // Skip special tokens in output (SentencePiece + Gemma + LLaMA 3)
    if (t.startsWith('<') && t.endsWith('>') && (
        t === '<s>' || t === '</s>' || t === '<unk>' || t === '<pad>'
        || t === '<start_of_turn>' || t === '<end_of_turn>'
        || t === '<bos>' || t === '<eos>'
        || t.startsWith('<|'))) return ''
    // Byte tokens: <0xHH> → actual byte
    const byteMatch = t.match(/^<0x([0-9A-Fa-f]{2})>$/)
    if (byteMatch) return String.fromCharCode(parseInt(byteMatch[1], 16))
    // GPT-2 BPE: map each unicode char back to its original byte, then decode UTF-8
    if (unicodeToByte) {
      const bytes = new Uint8Array(t.length)
      let byteLen = 0
      for (let i = 0; i < t.length; i++) {
        const cp = t.codePointAt(i)!
        const b = unicodeToByte.get(cp)
        if (b !== undefined) {
          bytes[byteLen++] = b
        }
      }
      return new TextDecoder().decode(bytes.subarray(0, byteLen))
    }
    // SentencePiece ▁ → space
    return t.replace(/\u2581/g, ' ')
  }

  function decode(ids: number[]): string {
    return ids.map(decodeToken).join('')
  }

  const endOfTurnTokenId = tokenToId.get('<end_of_turn>') ?? tokenToId.get('<|eot_id|>')

  return {
    vocabSize: vocab.length,
    bosTokenId,
    eosTokenId,
    endOfTurnTokenId,
    encode,
    encodeChat,
    decodeToken,
    decode,
  }
}
