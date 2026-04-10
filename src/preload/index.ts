import { contextBridge } from 'electron'

// IPC bridge — will be expanded in later phases
contextBridge.exposeInMainWorld('coral', {
  version: () => process.versions.electron
})
