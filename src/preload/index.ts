import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('opencoral', {
  version: () => process.versions.electron,

  // Network
  getNetworkState: () => ipcRenderer.invoke('opencoral:get-network-state'),

  // Model
  selectModel: () => ipcRenderer.invoke('opencoral:select-model'),
  loadModelPath: (filePath: string) => ipcRenderer.invoke('opencoral:load-model-path', filePath),
  getModel: () => ipcRenderer.invoke('opencoral:get-model'),

  // Block hosting
  startHosting: (blockStart: number, blockEnd: number) =>
    ipcRenderer.invoke('opencoral:start-hosting', blockStart, blockEnd),
  stopHosting: () => ipcRenderer.invoke('opencoral:stop-hosting'),
  getHostingState: () => ipcRenderer.invoke('opencoral:get-hosting-state'),

  // Coverage & inference
  checkCoverage: () => ipcRenderer.invoke('opencoral:check-coverage'),
  runInference: (prompt: string, nTokens: number) =>
    ipcRenderer.invoke('opencoral:run-inference', prompt, nTokens),

  // Hugging Face
  hfSearch: (query: string) => ipcRenderer.invoke('opencoral:hf-search', query),
  hfListFiles: (repoId: string) => ipcRenderer.invoke('opencoral:hf-list-files', repoId),
  hfDownload: (repoId: string, filenames: string[]) =>
    ipcRenderer.invoke('opencoral:hf-download', repoId, filenames),
  hfDownloadProgress: () => ipcRenderer.invoke('opencoral:hf-download-progress'),
  hfCancelDownload: () => ipcRenderer.invoke('opencoral:hf-cancel-download'),
  hfPreviewModel: (repoId: string, filename: string) =>
    ipcRenderer.invoke('opencoral:hf-preview-model', repoId, filename),
  hfEstimateBlocks: (filename: string, blockStart: number, blockEnd: number) =>
    ipcRenderer.invoke('opencoral:hf-estimate-blocks', filename, blockStart, blockEnd),
  hfDownloadPartial: (repoId: string, filename: string, blockStart: number, blockEnd: number) =>
    ipcRenderer.invoke('opencoral:hf-download-partial', repoId, filename, blockStart, blockEnd),
  hfDownloadShim: (repoId: string, filename: string) =>
    ipcRenderer.invoke('opencoral:hf-download-shim', repoId, filename),

  // Network model discovery
  discoverNetworkModels: () => ipcRenderer.invoke('opencoral:discover-network-models'),
  loadModelByHFIdentity: (repoId: string, hfFilename: string) =>
    ipcRenderer.invoke('opencoral:load-model-by-hf-identity', repoId, hfFilename),

  listLocalModels: () => ipcRenderer.invoke('opencoral:list-local-models'),
})
