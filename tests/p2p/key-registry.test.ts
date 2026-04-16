import { describe, it, expect, beforeAll, afterAll } from 'bun:test'
import { createOpenCoralNode, type OpenCoralNode } from '../../src/p2p/node'
import { loadOrCreateIdentity, type NodeIdentity } from '../../src/main/identity'
import {
  TrustedKeyStore,
  registerKeyRegistryHandler,
  registerWithBootstrap,
  lookupPeerKey,
} from '../../src/p2p/key-registry'
import { mkdtempSync, rmSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'

describe('TrustedKeyStore', () => {
  it('stores and retrieves keys', () => {
    const store = new TrustedKeyStore()
    const key = new Uint8Array(32).fill(0xAB)
    store.setKey('peer-1', key)
    expect(store.hasKey('peer-1')).toBe(true)
    expect(Array.from(store.getKey('peer-1')!)).toEqual(Array.from(key))
  })

  it('returns null for unknown peer', () => {
    const store = new TrustedKeyStore()
    expect(store.getKey('unknown')).toBeNull()
    expect(store.hasKey('unknown')).toBe(false)
  })
})

describe('key registry protocol', () => {
  let bootstrap: OpenCoralNode
  let peerNode: OpenCoralNode
  let lookupNode: OpenCoralNode
  let identityA: NodeIdentity
  let identityB: NodeIdentity
  let dirA: string
  let dirB: string
  let store: TrustedKeyStore

  beforeAll(async () => {
    dirA = mkdtempSync(join(tmpdir(), 'coral-keyreg-a-'))
    dirB = mkdtempSync(join(tmpdir(), 'coral-keyreg-b-'))
    identityA = await loadOrCreateIdentity(dirA)
    identityB = await loadOrCreateIdentity(dirB)

    bootstrap = await createOpenCoralNode()
    peerNode = await createOpenCoralNode()
    lookupNode = await createOpenCoralNode()

    await peerNode.libp2p.dial(bootstrap.libp2p.getMultiaddrs()[0])
    await lookupNode.libp2p.dial(bootstrap.libp2p.getMultiaddrs()[0])

    store = new TrustedKeyStore()
    await registerKeyRegistryHandler(bootstrap.libp2p, store)
  })

  afterAll(async () => {
    await bootstrap.stop()
    await peerNode.stop()
    await lookupNode.stop()
    rmSync(dirA, { recursive: true })
    rmSync(dirB, { recursive: true })
  })

  it('registerWithBootstrap stores key in bootstrap store', async () => {
    await registerWithBootstrap(peerNode.libp2p, bootstrap.libp2p.peerId, identityA)
    expect(store.hasKey(peerNode.peerId)).toBe(true)
    expect(Array.from(store.getKey(peerNode.peerId)!)).toEqual(Array.from(identityA.publicKey))
  })

  it('lookupPeerKey retrieves registered key', async () => {
    const key = await lookupPeerKey(lookupNode.libp2p, bootstrap.libp2p.peerId, peerNode.peerId)
    expect(key).not.toBeNull()
    expect(Array.from(key!)).toEqual(Array.from(identityA.publicKey))
  })

  it('lookupPeerKey returns null for unregistered peer', async () => {
    const key = await lookupPeerKey(lookupNode.libp2p, bootstrap.libp2p.peerId, 'nonexistent-peer')
    expect(key).toBeNull()
  })

  it('re-registration with different key is rejected', async () => {
    await expect(
      registerWithBootstrap(peerNode.libp2p, bootstrap.libp2p.peerId, identityB)
    ).rejects.toThrow()
  })
})
