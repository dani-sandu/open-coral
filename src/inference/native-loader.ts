import { createRequire } from 'module'
import { join } from 'path'

// createRequire gives us a require() that resolves from this file's directory
const _require = createRequire(import.meta.url)

function isPackaged(): boolean {
  try {
    return require('electron').app.isPackaged
  } catch {
    return false
  }
}

export interface CoralNative {
  hello(): string
  loadBlockRange(modelPath: string, blockStart: number, blockEnd: number, totalBlocks: number): number
  loadBlockRangeSharded(shardPaths: string[], blockStart: number, blockEnd: number, totalBlocks: number): number
  runForward(handle: number, input: Float32Array, nTokens: number): Float32Array
  freeBlockRange(handle: number): void
  embedTokens(handle: number, tokenIds: Int32Array): Float32Array
  projectToLogits(handle: number, hidden: Float32Array, nTokens: number): Float32Array
  getVocabSize(handle: number): number
  openSession(handle: number, maxLength: number): number
  closeSession(handle: number, sessionId: number): void
  sessionForward(handle: number, sessionId: number, input: Float32Array, nNewTokens: number): Float32Array
  loadVocab(path: string): number
  freeVocab(handle: number): void
  nativeTokenize(handle: number, text: string, addSpecial: boolean, parseSpecial: boolean): Int32Array
  nativeTokenToPiece(handle: number, tokenId: number): string
  nativeApplyChatTemplate(handle: number, userMessage: string): string
  nativeGetSpecialTokens(handle: number): { bosId: number; eosId: number; eotId: number; vocabSize: number }
}

let _cached: CoralNative | null = null

/** Resolve the absolute path to the native addon without loading it. */
export function getNativePath(): string {
  if (isPackaged()) {
    return join(process.resourcesPath, 'native', 'coral_native.node')
  }
  return new URL('../../native/build/Release/coral_native.node', import.meta.url)
    .pathname.replace(/^\/([A-Za-z]:)/, '$1')
}

export function getNative(): CoralNative {
  if (!_cached) {
    const addonPath = getNativePath()
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
