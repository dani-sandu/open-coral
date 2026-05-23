// planChainWithCandidates() triggers a DHT discovery + per-peer modelinfo cycle.
// Its result only populates response metadata, so a short TTL is safe and removes
// the cost from back-to-back inference requests.

const CHAIN_PLAN_TTL_MS = 5000

interface ChainPlanCacheEntry<T> {
  value: T
  expiresAt: number
}

const chainPlanCache = new Map<string, ChainPlanCacheEntry<unknown>>()

/**
 * Returns a cached chain plan for `repoId` if one was computed within the TTL,
 * otherwise runs `planFn`, caches its result, and returns it. Rejected promises
 * are not cached — the next call retries.
 */
export async function getChainPlanCached<T>(repoId: string, planFn: () => Promise<T>): Promise<T> {
  const now = Date.now()
  const cached = chainPlanCache.get(repoId) as ChainPlanCacheEntry<T> | undefined
  if (cached && cached.expiresAt > now) {
    return cached.value
  }
  const value = await planFn()
  chainPlanCache.set(repoId, { value, expiresAt: now + CHAIN_PLAN_TTL_MS })
  return value
}

/** Test-only handles. Do not use from production code. */
export const __testing__ = {
  clearChainPlanCache: (): void => { chainPlanCache.clear() },
  expireChainPlanCacheEntry: (repoId: string): void => {
    const e = chainPlanCache.get(repoId)
    if (e) e.expiresAt = 0
  },
}
