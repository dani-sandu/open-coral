import { readFileSync, writeFileSync, existsSync } from 'fs'
import { join } from 'path'
import { randomUUID } from 'crypto'

export interface ApiServerConfig {
  enabled: boolean
  port: number
  apiKey: string
}

export const DEFAULT_PORT = 39291

export function generateKey(): string {
  return `sk-local-${randomUUID()}`
}

export function loadConfig(userDataPath: string): ApiServerConfig {
  const filePath = join(userDataPath, 'api-config.json')
  if (existsSync(filePath)) {
    try {
      return JSON.parse(readFileSync(filePath, 'utf-8')) as ApiServerConfig
    } catch {
      // corrupted file — fall through to defaults
    }
  }
  const cfg: ApiServerConfig = { enabled: false, port: DEFAULT_PORT, apiKey: generateKey() }
  saveConfig(userDataPath, cfg)
  return cfg
}

export function saveConfig(userDataPath: string, cfg: ApiServerConfig): void {
  writeFileSync(join(userDataPath, 'api-config.json'), JSON.stringify(cfg, null, 2), 'utf-8')
}
