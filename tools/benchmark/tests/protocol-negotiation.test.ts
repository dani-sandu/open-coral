import { describe, it, expect } from 'bun:test'
import { mkdtempSync, rmSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { SimNetwork } from '../src/sim/sim-network'
import { loadOrCreateIdentity } from '../../../src/main/identity'
import {
  registerInferenceHandlerV3,
  registerInferenceHandlerV4,
  sendInferenceRequestNegotiated,
} from '../../../src/p2p/inference-protocol'
import { peerIdFromString } from '@libp2p/peer-id'

describe('protocol negotiation (V4 with V3 fallback)', () => {
  it('two V4-capable peers negotiate V4 and exchange fp16 (within tolerance)', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'coral-neg-v4-'))
    const identity = await loadOrCreateIdentity(dir)
    const nEmbd = 8
    const net = new SimNetwork({ latencyMeanMs: 1, latencyJitterMs: 0, modelBlocks: 1 })
    const client = await net.addNode()
    const worker = await net.addNode()
    await registerInferenceHandlerV3(worker.libp2p, async (input) => input, identity)
    await registerInferenceHandlerV4(worker.libp2p, async (input) => input, identity)

    const input = new Float32Array(nEmbd).fill(0.5)
    let info: { protocol: string; requestBytes: number } | undefined
    const out = await sendInferenceRequestNegotiated(
      client.libp2p, peerIdFromString(worker.peerId), input, 1, nEmbd, 'neg1', identity,
      { onWire: i => { info = i } },
    )
    expect(info?.protocol).toBe('/opencoral/inference/4.0.0')
    for (let i = 0; i < nEmbd; i++) expect(Math.abs(out[i] - 0.5)).toBeLessThanOrEqual(0.01)
    rmSync(dir, { recursive: true })
  })

  it('falls back to V3 (fp32, exact) when the peer only speaks V3', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'coral-neg-v3-'))
    const identity = await loadOrCreateIdentity(dir)
    const nEmbd = 8
    const net = new SimNetwork({ latencyMeanMs: 1, latencyJitterMs: 0, modelBlocks: 1 })
    const client = await net.addNode()
    const oldPeer = await net.addNode()
    await registerInferenceHandlerV3(oldPeer.libp2p, async (input) => input, identity)

    const input = new Float32Array(nEmbd).fill(0.123456)
    let info: { protocol: string; requestBytes: number } | undefined
    const out = await sendInferenceRequestNegotiated(
      client.libp2p, peerIdFromString(oldPeer.peerId), input, 1, nEmbd, 'neg2', identity,
      { onWire: i => { info = i } },
    )
    expect(info?.protocol).toBe('/opencoral/inference/3.0.0')
    for (let i = 0; i < nEmbd; i++) expect(out[i]).toBeCloseTo(0.123456, 5)
    rmSync(dir, { recursive: true })
  })

  it('reports a smaller requestBytes for V4 (fp16) than V3 (fp32) at the same shape', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'coral-neg-sz-'))
    const identity = await loadOrCreateIdentity(dir)
    const nEmbd = 256
    const net = new SimNetwork({ latencyMeanMs: 1, latencyJitterMs: 0, modelBlocks: 1 })
    const client = await net.addNode()
    const v4peer = await net.addNode()
    const v3peer = await net.addNode()
    await registerInferenceHandlerV4(v4peer.libp2p, async (i) => i, identity)
    await registerInferenceHandlerV3(v4peer.libp2p, async (i) => i, identity)
    await registerInferenceHandlerV3(v3peer.libp2p, async (i) => i, identity)

    const input = new Float32Array(nEmbd).fill(0.3)
    let v4info: any, v3info: any
    await sendInferenceRequestNegotiated(client.libp2p, peerIdFromString(v4peer.peerId), input, 1, nEmbd, 'a', identity, { onWire: i => { v4info = i } })
    await sendInferenceRequestNegotiated(client.libp2p, peerIdFromString(v3peer.peerId), input, 1, nEmbd, 'b', identity, { onWire: i => { v3info = i } })
    expect(v4info.protocol).toBe('/opencoral/inference/4.0.0')
    expect(v3info.protocol).toBe('/opencoral/inference/3.0.0')
    expect(v3info.requestBytes - v4info.requestBytes).toBeGreaterThanOrEqual(nEmbd * 2 - 8)
    rmSync(dir, { recursive: true })
  })
})
