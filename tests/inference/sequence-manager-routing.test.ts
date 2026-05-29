import { describe, it, expect } from 'bun:test'
import { SequenceManager, type ChainStepWithCandidates } from '../../src/inference/sequence-manager'

// Minimal fake node — the routing tests stub planChainWithCandidates, so the
// node/identity are never actually exercised by the refresh loop.
function makeMgr(): SequenceManager {
  const fakeNode = {
    peerId: 'self',
    multiaddrs: [] as string[],
    libp2p: { getPeers: () => [] } as unknown,
    stop: async () => {},
  }
  return new SequenceManager({
    node: fakeNode as never,
    localRunner: null,
    totalBlocks: 2,
    hiddenSize: 8,
    identity: {} as never,
  })
}

const FAKE_CHAIN: ChainStepWithCandidates[] = [
  { candidates: [{ peerId: 'p1', blockStart: 0, blockEnd: 1 }], blockStart: 0, blockEnd: 1 },
]

describe('SequenceManager routing-table background refresh', () => {
  it('getCachedChain() is null before any refresh', () => {
    expect(makeMgr().getCachedChain()).toBeNull()
  })

  it('refreshRoutingTable() populates the cached chain', async () => {
    const mgr = makeMgr()
    mgr.planChainWithCandidates = async () => FAKE_CHAIN
    await mgr.refreshRoutingTable()
    expect(mgr.getCachedChain()).toEqual(FAKE_CHAIN)
  })

  it('refreshRoutingTable() keeps the previous table when planning throws', async () => {
    const mgr = makeMgr()
    mgr.planChainWithCandidates = async () => FAKE_CHAIN
    await mgr.refreshRoutingTable()
    mgr.planChainWithCandidates = async () => { throw new Error('DHT down') }
    await mgr.refreshRoutingTable() // must not throw
    expect(mgr.getCachedChain()).toEqual(FAKE_CHAIN) // unchanged
  })

  it('startRoutingRefresh() refreshes immediately + on interval; stop halts it', async () => {
    const mgr = makeMgr()
    let calls = 0
    mgr.planChainWithCandidates = async () => { calls++; return FAKE_CHAIN }
    mgr.startRoutingRefresh(20)
    await new Promise(r => setTimeout(r, 75))
    expect(calls).toBeGreaterThanOrEqual(2)
    expect(mgr.getCachedChain()).toEqual(FAKE_CHAIN)
    mgr.stopRoutingRefresh()
    const atStop = calls
    await new Promise(r => setTimeout(r, 60))
    expect(calls).toBe(atStop)
  })

  it('startRoutingRefresh() is idempotent (no second timer on double-start)', async () => {
    const mgr = makeMgr()
    let calls = 0
    mgr.planChainWithCandidates = async () => { calls++; return FAKE_CHAIN }
    mgr.startRoutingRefresh(20)
    mgr.startRoutingRefresh(20) // ignored — already running
    await new Promise(r => setTimeout(r, 75))
    mgr.stopRoutingRefresh()
    expect(calls).toBeLessThanOrEqual(6)
  })

  it('stopRoutingRefresh() is safe to call when never started', () => {
    expect(() => makeMgr().stopRoutingRefresh()).not.toThrow()
  })
})
