import { readFileSync, writeFileSync, existsSync, unlinkSync } from 'fs'
import { join } from 'path'
import { homedir } from 'os'

// These are the env keys we own when Claude integration is active.
// All others (NODE_EXTRA_CA_CERTS, etc.) are left untouched.
const API_ENV_KEYS = [
  'ANTHROPIC_BASE_URL',
  'ANTHROPIC_API_KEY',
  'ANTHROPIC_AUTH_TOKEN',
  'ANTHROPIC_MODEL',
  'ANTHROPIC_DEFAULT_OPUS_MODEL',
  'ANTHROPIC_DEFAULT_SONNET_MODEL',
  'ANTHROPIC_DEFAULT_HAIKU_MODEL',
  'ANTHROPIC_CUSTOM_HEADERS',
] as const

type ApiEnvKey = typeof API_ENV_KEYS[number]
type EnvBackup = Partial<Record<ApiEnvKey, string>>

function claudeSettingsPath(): string {
  return join(homedir(), '.claude', 'settings.json')
}

function backupPath(userDataPath: string): string {
  return join(userDataPath, 'claude-env-backup.json')
}

export function isEnabledInClaude(userDataPath: string): boolean {
  return existsSync(backupPath(userDataPath))
}

export function enableInClaude(userDataPath: string, port: number, apiKey: string): void {
  const settingsPath = claudeSettingsPath()

  let settings: Record<string, unknown> = {}
  if (existsSync(settingsPath)) {
    try {
      settings = JSON.parse(readFileSync(settingsPath, 'utf-8'))
    } catch {
      settings = {}
    }
  }

  const env = (settings.env ?? {}) as Record<string, string>

  // Save current values of all API-related keys (including undefined = absent)
  const backup: EnvBackup = {}
  for (const key of API_ENV_KEYS) {
    if (Object.prototype.hasOwnProperty.call(env, key)) {
      backup[key] = env[key]
    }
  }
  writeFileSync(backupPath(userDataPath), JSON.stringify(backup, null, 2), 'utf-8')

  // Remove all API keys, then install ours
  for (const key of API_ENV_KEYS) {
    delete env[key]
  }
  // ANTHROPIC_AUTH_TOKEN sends Authorization: Bearer <token> which matches our server's auth check
  env['ANTHROPIC_BASE_URL'] = `http://localhost:${port}`
  env['ANTHROPIC_AUTH_TOKEN'] = apiKey

  settings.env = env
  writeFileSync(settingsPath, JSON.stringify(settings, null, 2), 'utf-8')
}

export function disableInClaude(userDataPath: string): void {
  const bPath = backupPath(userDataPath)
  if (!existsSync(bPath)) return

  const backup: EnvBackup = JSON.parse(readFileSync(bPath, 'utf-8'))

  const settingsPath = claudeSettingsPath()
  let settings: Record<string, unknown> = {}
  if (existsSync(settingsPath)) {
    try {
      settings = JSON.parse(readFileSync(settingsPath, 'utf-8'))
    } catch {
      settings = {}
    }
  }

  const env = (settings.env ?? {}) as Record<string, string>

  // Remove keys we installed
  for (const key of API_ENV_KEYS) {
    delete env[key]
  }

  // Restore original values
  for (const [key, value] of Object.entries(backup) as [ApiEnvKey, string][]) {
    env[key] = value
  }

  settings.env = env
  writeFileSync(settingsPath, JSON.stringify(settings, null, 2), 'utf-8')
  unlinkSync(bPath)
}
