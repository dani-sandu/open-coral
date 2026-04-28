// tests/main/session-store.test.ts
import { describe, it, expect, beforeEach, afterEach } from 'bun:test'
import { promises as fs } from 'fs'
import { join } from 'path'
import { tmpdir } from 'os'
import { SessionStore } from '../../src/main/session-store'

let dir: string

beforeEach(async () => {
  dir = await fs.mkdtemp(join(tmpdir(), 'opencoral-store-'))
})

afterEach(async () => {
  await fs.rm(dir, { recursive: true, force: true })
})

describe('SessionStore', () => {
  it('round-trips a session through save/load', async () => {
    const store = new SessionStore(dir)
    await store.init()

    const session = {
      schemaVersion: 1 as const,
      id: 's1',
      title: 'Hello',
      createdAt: 1000,
      updatedAt: 2000,
      messages: [
        { id: 'm1', role: 'user' as const, text: 'hi', timestamp: 1500 },
      ],
    }

    await store.save(session)
    await store.flush()  // force pending debounced write

    const loaded = await store.load('s1')
    expect(loaded).toEqual(session)
  })

  it('survives a partial write (no .tmp visible after flush, atomic rename)', async () => {
    const store = new SessionStore(dir)
    await store.init()

    await store.save({
      schemaVersion: 1, id: 's1', title: 't', createdAt: 1, updatedAt: 1, messages: [],
    })
    await store.flush()

    const files = await fs.readdir(dir)
    // After flush, no orphaned .tmp files should remain.
    expect(files.some(f => f.endsWith('.tmp'))).toBe(false)
    expect(files.some(f => f === 's1.json')).toBe(true)
  })

  it('marks corrupt files as corrupt in the index without deleting them', async () => {
    const store = new SessionStore(dir)
    await store.init()

    // Write a valid session, then overwrite with garbage to simulate corruption.
    await store.save({
      schemaVersion: 1, id: 's2', title: 't', createdAt: 1, updatedAt: 1, messages: [],
    })
    await store.flush()
    await fs.writeFile(join(dir, 's2.json'), '{not valid json', 'utf-8')

    const loaded = await store.load('s2')
    expect(loaded).toBeNull()

    const list = await store.list()
    const entry = list.find(s => s.id === 's2')
    expect(entry).toBeDefined()
    expect(entry!.corrupt).toBe(true)

    // File still exists — never auto-delete user data.
    const files = await fs.readdir(dir)
    expect(files).toContain('s2.json')
  })

  it('coalesces rapid saves into one write via debouncing', async () => {
    const store = new SessionStore(dir, { debounceMs: 50 })
    await store.init()

    // Three saves in quick succession — only the last should hit disk after flush.
    await store.save({
      schemaVersion: 1, id: 's3', title: 'v1', createdAt: 1, updatedAt: 1, messages: [],
    })
    await store.save({
      schemaVersion: 1, id: 's3', title: 'v2', createdAt: 1, updatedAt: 2, messages: [],
    })
    await store.save({
      schemaVersion: 1, id: 's3', title: 'v3', createdAt: 1, updatedAt: 3, messages: [],
    })
    await store.flush()

    const loaded = await store.load('s3')
    expect(loaded?.title).toBe('v3')
    expect(loaded?.updatedAt).toBe(3)
  })

  it('does not re-create deleted sessions after pending debounced write', async () => {
    const store = new SessionStore(dir, { debounceMs: 50 })
    await store.init()

    const s = {
      schemaVersion: 1 as const,
      id: 's4',
      title: 'temp',
      createdAt: 1000,
      updatedAt: 2000,
      messages: [],
    }

    // Schedule a debounced write.
    await store.save(s)

    // Immediately delete before the timer fires.
    await store.delete(s.id)

    // Wait longer than the debounce interval to ensure any stale timer callback fires.
    await new Promise(resolve => setTimeout(resolve, 100))

    // Verify the file was not re-created.
    const files = await fs.readdir(dir)
    expect(files).not.toContain('s4.json')

    // Verify the session is not in the index.
    const list = await store.list()
    expect(list.find(x => x.id === s.id)).toBeUndefined()
  })
})
