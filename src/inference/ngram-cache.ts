/**
 * N-gram lookup cache for speculative decoding.
 *
 * Uses the ngram-simple strategy: scan all prior tokens for matching n-grams
 * and return the tokens that followed the match as draft candidates.
 */
export class NgramCache {
  private readonly ngramSize: number
  private readonly draftMax: number
  private readonly table = new Map<string, number[]>()

  constructor(ngramSize: number, draftMax: number) {
    this.ngramSize = ngramSize
    this.draftMax = draftMax
  }

  buildFromTokens(tokens: number[]): void {
    this.table.clear()
    for (let i = 0; i <= tokens.length - this.ngramSize; i++) {
      const key = this.makeKey(tokens, i, this.ngramSize)
      const contStart = i + this.ngramSize
      const contEnd = Math.min(contStart + this.draftMax, tokens.length)
      if (contStart < tokens.length) {
        this.table.set(key, tokens.slice(contStart, contEnd))
      }
    }
  }

  addToken(_token: number, allTokens: number[]): void {
    const len = allTokens.length
    if (len < this.ngramSize + 1) return
    for (let lookback = 1; lookback <= this.draftMax && lookback < len - this.ngramSize + 1; lookback++) {
      const ngramStart = len - this.ngramSize - lookback
      if (ngramStart < 0) break
      const key = this.makeKey(allTokens, ngramStart, this.ngramSize)
      const contStart = ngramStart + this.ngramSize
      const contEnd = Math.min(contStart + this.draftMax, len)
      this.table.set(key, allTokens.slice(contStart, contEnd))
    }
  }

  lookup(context: number[]): number[] {
    if (context.length < this.ngramSize) return []
    const keyStart = context.length - this.ngramSize
    const key = this.makeKey(context, keyStart, this.ngramSize)
    return this.table.get(key) ?? []
  }

  private makeKey(tokens: number[], start: number, len: number): string {
    let key = ''
    for (let i = 0; i < len; i++) {
      if (i > 0) key += ','
      key += tokens[start + i]
    }
    return key
  }
}
