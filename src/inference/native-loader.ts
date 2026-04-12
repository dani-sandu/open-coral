import { createRequire } from 'module'

// createRequire gives us a require() that resolves from this file's directory
const _require = createRequire(import.meta.url)

export interface CoralNative {
  hello(): string
  loadBlockRange(modelPath: string, blockStart: number, blockEnd: number, totalBlocks: number): number
  runForward(handle: number, input: Float32Array, nTokens: number): Float32Array
  freeBlockRange(handle: number): void
  embedTokens(handle: number, tokenIds: Int32Array): Float32Array
  projectToLogits(handle: number, hidden: Float32Array, nTokens: number): Float32Array
  getVocabSize(handle: number): number
  openSession(handle: number, maxLength: number): number
  closeSession(handle: number, sessionId: number): void
  sessionForward(handle: number, sessionId: number, input: Float32Array, nNewTokens: number): Float32Array
}

let _cached: CoralNative | null = null

export function getNative(): CoralNative {
  if (!_cached) {
    const addonPath = new URL('../../native/build/Release/coral_native.node', import.meta.url)
      .pathname.replace(/^\/([A-Za-z]:)/, '$1')
    try {
      _cached = _require(addonPath) as CoralNative
    } catch {
      throw new Error(
        `coral_native.node not found at ${addonPath} — run \`npm run build:native\` first`
      )
    }
  }
  return _cached
}
