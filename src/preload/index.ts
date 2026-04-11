import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('coral', {
  version: () => process.versions.electron,

  // Network
  getNetworkState: () => ipcRenderer.invoke('coral:get-network-state'),

  // Model
  selectModel: () => ipcRenderer.invoke('coral:select-model'),
  loadModelPath: (filePath: string) => ipcRenderer.invoke('coral:load-model-path', filePath),
  getModel: () => ipcRenderer.invoke('coral:get-model'),

  // Block hosting
  startHosting: (blockStart: number, blockEnd: number) =>
    ipcRenderer.invoke('coral:start-hosting', blockStart, blockEnd),
  stopHosting: () => ipcRenderer.invoke('coral:stop-hosting'),
  getHostingState: () => ipcRenderer.invoke('coral:get-hosting-state'),

  // Inference demo
  runInferenceDemo: (nTokens: number) =>
    ipcRenderer.invoke('coral:run-inference-demo', nTokens),

  // Hugging Face
  hfSearch: (query: string) => ipcRenderer.invoke('coral:hf-search', query),
  hfListFiles: (repoId: string) => ipcRenderer.invoke('coral:hf-list-files', repoId),
  hfDownload: (repoId: string, filename: string) =>
    ipcRenderer.invoke('coral:hf-download', repoId, filename),
  hfDownloadProgress: () => ipcRenderer.invoke('coral:hf-download-progress'),
  hfCancelDownload: () => ipcRenderer.invoke('coral:hf-cancel-download'),
  hfPreviewModel: (repoId: string, filename: string) =>
    ipcRenderer.invoke('coral:hf-preview-model', repoId, filename),
  hfEstimateBlocks: (blockStart: number, blockEnd: number) =>
    ipcRenderer.invoke('coral:hf-estimate-blocks', blockStart, blockEnd),
  hfDownloadPartial: (repoId: string, filename: string, blockStart: number, blockEnd: number) =>
    ipcRenderer.invoke('coral:hf-download-partial', repoId, filename, blockStart, blockEnd),
})
