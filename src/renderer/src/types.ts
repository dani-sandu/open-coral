export interface ModelInfo {
  path: string
  architecture: string
  totalBlocks: number
  hiddenSize: number
  headCount: number
  fileSizeBytes: number
  /** HuggingFace repo ID. Present only for HF downloads. */
  repoId?: string
  /** HuggingFace filename. Present only for HF downloads. */
  hfFilename?: string
  /** True when loaded from a shim GGUF (no block tensors) */
  shimOnly?: boolean
  shardFiles?: string[]
}

export interface HostingState {
  modelPath: string
  blockStart: number
  blockEnd: number
  totalBlocks: number
  hiddenSize: number
}

export interface BlockCoverage {
  blockStart: number
  blockEnd: number
  /** 'local' for the local BlockRunner, or a remote peer ID string */
  peerId: string
  multiaddrs: string[]
}

export interface CoverageReport {
  totalBlocks: number
  covered: BlockCoverage[]
  missing: number[]
  complete: boolean
  suggestion?: { start: number; end: number }
}

// NOTE: Keep in sync with InferenceResult in src/main/block-host.ts.
// Duplication is intentional: main and renderer processes cannot share TS source directly.
export interface InferenceResult {
  prompt: string
  generatedText: string
  generatedTokens: number
  nEmbd: number
  chainSteps: { peerId: string; blockStart: number; blockEnd: number; durationMs: number }[]
  totalDurationMs: number
  specDraftTokens?: number
  specAcceptedTokens?: number
  specAcceptanceRate?: number
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
  result?: InferenceResult
  error?: string
  timestamp: number
}

export interface ChatSession {
  id: string
  title: string
  messages: ChatMessage[]
  createdAt: number
  updatedAt: number
}

export interface NetworkState {
  localPeerId: string
  localMultiaddrs: string[]
  localBlocks: { start: number; end: number }[]
  peers: { peerId: string; multiaddrs: string[]; blockRanges: { start: number; end: number }[]; isLocal: boolean; connected: boolean; modelInfo?: PeerModelInfo }[]
  connections: { from: string; to: string }[]
  timestamp: number
}

export interface HFModelResult {
  id: string
  author: string
  modelId: string
  likes: number
  downloads: number
  tags: string[]
  lastModified: string
}

export interface HFFileInfo {
  rfilename: string
  size: number
}

// NOTE: Keep in sync with ShardSet in src/inference/shard-utils.ts.
// Duplicated intentionally — the renderer cannot import main-process modules.
export interface ShardSet {
  canonical: string
  shardFiles: string[]
  totalShards: number
  combinedSize: number
}

export interface DownloadProgress {
  file: string
  downloadedBytes: number
  totalBytes: number
  percent: number
  done: boolean
  error?: string
  localPath?: string
  currentShard?: number
  totalShards?: number
}

export interface HFModelPreview {
  architecture: string
  totalBlocks: number
  hiddenSize: number
  headCount: number
  repoId: string
  filename: string
}

export interface BlockEstimate {
  partialSize: number
  fullSize: number
  savedPercent: number
}

export interface PeerModelInfo {
  repoId: string
  hfFilename: string
  blockStart: number
  blockEnd: number
  totalBlocks: number
  hiddenSize: number
  architecture: string
}

export interface NetworkModelEntry {
  repoId: string
  hfFilename: string
  totalBlocks: number
  hiddenSize: number
  architecture: string
  coveredBlocks: number
  complete: boolean
  peers: { peerId: string; blockStart: number; blockEnd: number }[]
}

export interface LocalModelEntry {
  path: string
  filename: string
  architecture: string
  totalBlocks: number
  hiddenSize: number
  headCount: number
  fileSizeBytes: number
  repoId?: string
  hfFilename?: string
  blockStart: number | null
  blockEnd: number | null
  shardFiles?: string[]
}

export interface SessionSummary {
  id: string
  title: string
  createdAt: number
  updatedAt: number
  messageCount: number
  corrupt?: boolean
}

export interface SessionPhaseEvent {
  sessionId: string
  phase: 'planning' | 'opening-remote-kv' | 'prefilling' | 'ready' | 'error'
  prefilledTokens?: number
  totalTokens?: number
  error?: string
}

export interface SessionInvalidationEvent {
  sessionId: string
  reason: 'peer-drop' | 'model-change'
}

export type Unsubscribe = () => void

declare global {
  interface Window {
    opencoral: {
      version: () => string
      getNetworkState: () => Promise<NetworkState | null>
      selectModel: () => Promise<ModelInfo | null>
      loadModelPath: (filePath: string) => Promise<ModelInfo>
      getModel: () => Promise<ModelInfo | null>
      startHosting: (blockStart: number, blockEnd: number) => Promise<void>
      stopHosting: () => Promise<void>
      getHostingState: () => Promise<HostingState | null>
      checkCoverage: () => Promise<CoverageReport>
      runInference: (prompt: string, maxTokens: number) => Promise<InferenceResult>
      hfSearch: (query: string) => Promise<HFModelResult[]>
      hfListFiles: (repoId: string) => Promise<HFFileInfo[]>
      hfDownload: (repoId: string, filenames: string[]) => Promise<string>
      hfDownloadProgress: () => Promise<DownloadProgress | null>
      hfCancelDownload: () => Promise<void>
      hfPreviewModel: (repoId: string, filename: string) => Promise<HFModelPreview>
      hfEstimateBlocks: (filename: string, blockStart: number, blockEnd: number) => Promise<BlockEstimate>
      hfDownloadPartial: (repoId: string, filename: string, blockStart: number, blockEnd: number) => Promise<string>
      hfDownloadShim: (repoId: string, filename: string) => Promise<string>
      discoverNetworkModels: () => Promise<NetworkModelEntry[]>
      loadModelByHFIdentity: (repoId: string, hfFilename: string) => Promise<ModelInfo>
      listLocalModels: () => Promise<LocalModelEntry[]>
      // Chat sessions
      listSessions: () => Promise<SessionSummary[]>
      getSession: (id: string) => Promise<ChatSession | null>
      createSession: () => Promise<SessionSummary>
      sendTurn: (sessionId: string, userText: string, maxTokens: number) => Promise<{ generatedText: string; trace: InferenceResult }>
      deleteSession: (id: string) => Promise<void>
      renameSession: (id: string, title: string) => Promise<void>
      onSessionUpdated: (handler: (s: SessionSummary) => void) => Unsubscribe
      onSessionDeleted: (handler: (id: string) => void) => Unsubscribe
      onSessionPhase: (handler: (e: SessionPhaseEvent) => void) => Unsubscribe
      onSessionInvalidated: (handler: (e: SessionInvalidationEvent) => void) => Unsubscribe
    }
  }
}
