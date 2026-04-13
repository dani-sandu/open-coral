import { describe, it, expect, beforeEach, afterEach } from 'bun:test'
import { mkdtempSync, rmSync } from 'fs'
import { join } from 'path'
import { tmpdir } from 'os'
import { loadOrCreateIdentity, type NodeIdentity } from '../../src/main/identity'

let tmpDir: string

beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), 'coral-identity-test-'))
})

afterEach(() => {
  rmSync(tmpDir, { recursive: true })
})

describe('loadOrCreateIdentity', () => {
  it('creates a new identity when no file exists', async () => {
    const id: NodeIdentity = await loadOrCreateIdentity(tmpDir)
    expect(id.privateKey).toBeInstanceOf(Uint8Array)
    expect(id.publicKey).toBeInstanceOf(Uint8Array)
    expect(id.privateKey.byteLength).toBeGreaterThan(0)
    expect(id.publicKey.byteLength).toBeGreaterThan(0)
  })

  it('returns the same identity on second call (persisted)', async () => {
    const first = await loadOrCreateIdentity(tmpDir)
    const second = await loadOrCreateIdentity(tmpDir)
    expect(Buffer.from(first.publicKey).toString('hex'))
      .toBe(Buffer.from(second.publicKey).toString('hex'))
  })

  it('produces distinct key pairs for different directories', async () => {
    const tmpDir2 = mkdtempSync(join(tmpdir(), 'coral-identity-test2-'))
    try {
      const a = await loadOrCreateIdentity(tmpDir)
      const b = await loadOrCreateIdentity(tmpDir2)
      expect(Buffer.from(a.publicKey).toString('hex'))
        .not.toBe(Buffer.from(b.publicKey).toString('hex'))
    } finally {
      rmSync(tmpDir2, { recursive: true })
    }
  })
})
