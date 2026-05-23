import { describe, it, expect, beforeEach } from 'bun:test'
import { getChainPlanCached, __testing__ } from '../../src/main/chain-plan-cache'

describe('chain plan cache', () => {
  beforeEach(() => {
    __testing__.clearChainPlanCache()
  })

  it('returns cached value within TTL', async () => {
    let planCalls = 0
    const planFn = async () => { planCalls++; return [{ candidates: [{ peerId: 'p', blockStart: 0, blockEnd: 0 }], blockStart: 0, blockEnd: 0 }] }

    const r1 = await getChainPlanCached('repo-A', planFn)
    const r2 = await getChainPlanCached('repo-A', planFn)
    expect(planCalls).toBe(1)
    expect(r1).toEqual(r2)
  })

  it('refetches after TTL expiry', async () => {
    let planCalls = 0
    const planFn = async () => { planCalls++; return [] as any[] }

    await getChainPlanCached('repo-B', planFn)
    __testing__.expireChainPlanCacheEntry('repo-B')
    await getChainPlanCached('repo-B', planFn)
    expect(planCalls).toBe(2)
  })

  it('keeps separate cache entries per repoId', async () => {
    let planCalls = 0
    const planFn = async () => { planCalls++; return [] as any[] }
    await getChainPlanCached('repo-X', planFn)
    await getChainPlanCached('repo-Y', planFn)
    expect(planCalls).toBe(2)
  })

  it('does not cache errors', async () => {
    let attempts = 0
    const planFn = async () => {
      attempts++
      if (attempts === 1) throw new Error('DHT timeout')
      return [] as any[]
    }
    await expect(getChainPlanCached('repo-Z', planFn)).rejects.toThrow('DHT timeout')
    await getChainPlanCached('repo-Z', planFn)
    expect(attempts).toBe(2)
  })

  it('concurrent cache misses each compute (no in-flight de-duplication)', async () => {
    // Two near-simultaneous calls for the same repo before either resolves.
    // The cache keys on settled values, not in-flight promises, so both miss
    // and both compute. Pinned with toBe(2) so adding in-flight dedup later is
    // a deliberate, visible test change.
    let planCalls = 0
    const planFn = async () => {
      planCalls++
      await new Promise(r => setTimeout(r, 10))
      return [] as any[]
    }
    await Promise.all([
      getChainPlanCached('repo-C', planFn),
      getChainPlanCached('repo-C', planFn),
    ])
    expect(planCalls).toBe(2)
  })
})
