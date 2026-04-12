import { describe, it, expect } from 'bun:test'
import { DiscoveredModels } from '../../src/p2p/discovered-models'
import type { PeerModelInfoPayload } from '../../src/p2p/model-announce'

const MODEL_A: PeerModelInfoPayload = {
  repoId: 'org/ModelA-GGUF',
  hfFilename: 'ModelA-Q4_K_M.gguf',
  blockStart: 0,
  blockEnd: 15,
  totalBlocks: 32,
  hiddenSize: 4096,
  architecture: 'llama',
}

const MODEL_B: PeerModelInfoPayload = {
  repoId: 'org/ModelB-GGUF',
  hfFilename: 'ModelB-Q4_K_M.gguf',
  blockStart: 0,
  blockEnd: 31,
  totalBlocks: 32,
  hiddenSize: 2048,
  architecture: 'llama',
}

describe('DiscoveredModels', () => {
  it('starts empty', () => {
    const reg = new DiscoveredModels()
    expect(reg.aggregate()).toHaveLength(0)
  })

  it('adds a peer and aggregates into a NetworkModelEntry', () => {
    const reg = new DiscoveredModels()
    reg.update('peer1', MODEL_A)
    const entries = reg.aggregate()
    expect(entries).toHaveLength(1)
    const entry = entries[0]
    expect(entry.coveredBlocks).toBe(16) // blocks 0..15 = 16 blocks
    expect(entry.complete).toBe(false)   // 16/32
    expect(entry.peers).toHaveLength(1)
    expect(entry.peers[0].peerId).toBe('peer1')
  })

  it('two peers covering different halves of the same model → complete', () => {
    const reg = new DiscoveredModels()
    reg.update('peer1', { ...MODEL_A, blockStart: 0, blockEnd: 15 })
    reg.update('peer2', { ...MODEL_A, blockStart: 16, blockEnd: 31 })
    const entries = reg.aggregate()
    expect(entries).toHaveLength(1)
    const entry = entries[0]
    expect(entry.coveredBlocks).toBe(32)
    expect(entry.complete).toBe(true)
    expect(entry.peers).toHaveLength(2)
  })

  it('two peers covering overlapping ranges → correct union count', () => {
    const reg = new DiscoveredModels()
    reg.update('peer1', { ...MODEL_A, blockStart: 0, blockEnd: 20 })
    reg.update('peer2', { ...MODEL_A, blockStart: 10, blockEnd: 31 })
    const entries = reg.aggregate()
    expect(entries).toHaveLength(1)
    const entry = entries[0]
    expect(entry.coveredBlocks).toBe(32)
    expect(entry.complete).toBe(true)
  })

  it('two different models → two entries', () => {
    const reg = new DiscoveredModels()
    reg.update('peer1', MODEL_A)
    reg.update('peer2', MODEL_B)
    const entries = reg.aggregate()
    expect(entries).toHaveLength(2)
  })

  it('remove() deletes peer from registry', () => {
    const reg = new DiscoveredModels()
    reg.update('peer1', MODEL_A)
    reg.remove('peer1')
    expect(reg.aggregate()).toHaveLength(0)
  })

  it('updating existing peer replaces its old entry', () => {
    const reg = new DiscoveredModels()
    reg.update('peer1', { ...MODEL_A, blockStart: 0, blockEnd: 7 })
    reg.update('peer1', { ...MODEL_A, blockStart: 0, blockEnd: 15 })
    const entries = reg.aggregate()
    expect(entries).toHaveLength(1)
    const entry = entries[0]
    expect(entry.coveredBlocks).toBe(16)
    expect(entry.peers).toHaveLength(1)
  })
})
