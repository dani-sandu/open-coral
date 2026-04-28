/**
 * N-gram lookup cache for speculative decoding.
 *
 * Uses the ngram-simple strategy: scan all prior tokens for matching n-grams
 * and return the tokens that followed the match as draft candidates.
 */
export class NgramCache {
  private readonly ngramSize: number
  private readonly draftMax: number
  private readonly table = new Map<number, number[]>()

  constructor(ngramSize = 4, draftMax = 5) {
    this.ngramSize = ngramSize
    this.draftMax = draftMax
  }

  buildFromTokens(tokens: number[]): void {
    this.table.clear()
    for (let i = 0; i <= tokens.length - this.ngramSize; i++) {
      const key = this.hashKey(tokens, i, this.ngramSize)
      const contStart = i + this.ngramSize
      const contEnd = Math.min(contStart + this.draftMax, tokens.length)
      if (contStart < tokens.length) {
        this.table.set(key, tokens.slice(contStart, contEnd))
      }
    }
  }

  addToken(_token: number, allTokens: number[]): void {
    // _token is already the last element of allTokens; kept for call-site symmetry
    const len = allTokens.length
    if (len < this.ngramSize + 1) return
    for (let lookback = 1; lookback <= this.draftMax && lookback < len - this.ngramSize + 1; lookback++) {
      const ngramStart = len - this.ngramSize - lookback
      if (ngramStart < 0) break
      const key = this.hashKey(allTokens, ngramStart, this.ngramSize)
      const contStart = ngramStart + this.ngramSize
      const contEnd = Math.min(contStart + this.draftMax, len)
      this.table.set(key, allTokens.slice(contStart, contEnd))
    }
  }

  lookup(context: number[]): number[] {
    if (context.length < this.ngramSize) return []
    const key = this.hashKey(context, context.length - this.ngramSize, this.ngramSize)
    return this.table.get(key) ?? []
  }

  // 32-bit FNV-1a. Collisions are rare but possible for large vocabularies (e.g. 128K);
  // the consequence is a wrong draft candidate, which speculative verification rejects —
  // safe, not catastrophic.
  private hashKey(tokens: number[], start: number, len: number): number {
    let h = 2166136261  // FNV-1a 32-bit offset basis
    for (let i = 0; i < len; i++) {
      const id = tokens[start + i]
      h ^= id & 0xff;          h = Math.imul(h, 16777619)
      h ^= (id >>> 8) & 0xff;  h = Math.imul(h, 16777619)
      h ^= (id >>> 16) & 0xff; h = Math.imul(h, 16777619)
      h ^= (id >>> 24) & 0xff; h = Math.imul(h, 16777619)
    }
    return h >>> 0
  }
}
