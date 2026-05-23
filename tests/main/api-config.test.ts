import { describe, it, expect, beforeEach, afterEach } from 'bun:test'
import { mkdirSync, rmSync, existsSync } from 'fs'
import { join } from 'path'
import { tmpdir } from 'os'
import { loadConfig, saveConfig, generateKey } from '../../src/main/api-config'

let tmpDir: string

beforeEach(() => {
  tmpDir = join(tmpdir(), `api-config-test-${Date.now()}`)
  mkdirSync(tmpDir, { recursive: true })
})

afterEach(() => {
  rmSync(tmpDir, { recursive: true, force: true })
})

describe('generateKey', () => {
  it('returns a string starting with sk-local-', () => {
    expect(generateKey()).toMatch(/^sk-local-[0-9a-f-]{36}$/)
  })

  it('returns a different value each call', () => {
    expect(generateKey()).not.toBe(generateKey())
  })
})

describe('loadConfig', () => {
  it('creates a default config when no file exists', () => {
    const cfg = loadConfig(tmpDir)
    expect(cfg.enabled).toBe(false)
    expect(cfg.port).toBe(39291)
    expect(cfg.apiKey).toMatch(/^sk-local-/)
  })

  it('persists the generated key on first load', () => {
    const cfg1 = loadConfig(tmpDir)
    const cfg2 = loadConfig(tmpDir)
    expect(cfg1.apiKey).toBe(cfg2.apiKey)
  })

  it('reads back a saved config', () => {
    saveConfig(tmpDir, { enabled: true, port: 12345, apiKey: 'sk-local-test' })
    const cfg = loadConfig(tmpDir)
    expect(cfg.enabled).toBe(true)
    expect(cfg.port).toBe(12345)
    expect(cfg.apiKey).toBe('sk-local-test')
  })
})

describe('saveConfig', () => {
  it('writes a json file to the given directory', () => {
    saveConfig(tmpDir, { enabled: false, port: 39291, apiKey: 'sk-local-x' })
    expect(existsSync(join(tmpDir, 'api-config.json'))).toBe(true)
  })
})
