import { describe, it, expect } from 'bun:test'
import { ThinkingBudget, hasOpenThinkBlock, THINKING_TOKEN_BUDGET } from '../../src/inference/thinking-budget'

const TT = { open: 1, close: 2 }

describe('hasOpenThinkBlock', () => {
  it('returns false when thinkTokens is undefined', () => {
    expect(hasOpenThinkBlock([1, 2, 3], undefined)).toBe(false)
  })
  it('returns true for an unclosed <think>', () => {
    expect(hasOpenThinkBlock([9, 1, 5, 5], TT)).toBe(true)
  })
  it('returns false for a balanced <think></think>', () => {
    expect(hasOpenThinkBlock([1, 5, 2, 9], TT)).toBe(false)
  })
  it('returns false when no think tokens present', () => {
    expect(hasOpenThinkBlock([9, 5, 5], TT)).toBe(false)
  })
})

describe('ThinkingBudget', () => {
  it('non-thinking model: stays in answer phase, plain cap', () => {
    const b = new ThinkingBudget(undefined, 100, 3, true)
    expect(b.thinking).toBe(false)
    b.count(5); b.count(5)
    expect(b.remaining()).toBe(1)
    expect(b.shouldContinue()).toBe(true)
    b.count(5)
    expect(b.shouldContinue()).toBe(false)
  })

  it('thinking tokens count against the thinking budget, not the answer cap', () => {
    const b = new ThinkingBudget(TT, 10, 3, true)
    expect(b.thinking).toBe(true)
    b.count(5); b.count(5); b.count(5); b.count(5)
    expect(b.remaining()).toBe(6)
    expect(b.shouldContinue()).toBe(true)
  })

  it('a generated </think> switches to the answer phase', () => {
    const b = new ThinkingBudget(TT, 10, 2, true)
    b.count(5)
    b.count(TT.close)
    expect(b.thinking).toBe(false)
    expect(b.remaining()).toBe(2)
    b.count(7); b.count(8)
    expect(b.shouldContinue()).toBe(false)
  })

  it('needsForceClose is true once the thinking budget is spent', () => {
    const b = new ThinkingBudget(TT, 3, 5, true)
    b.count(5); b.count(5)
    expect(b.needsForceClose()).toBe(false)
    b.count(5)
    expect(b.needsForceClose()).toBe(true)
    expect(b.shouldContinue()).toBe(true)
  })

  it('forceClose switches to the answer phase', () => {
    const b = new ThinkingBudget(TT, 3, 5, true)
    b.count(5); b.count(5); b.count(5)
    expect(b.needsForceClose()).toBe(true)
    b.forceClose()
    expect(b.thinking).toBe(false)
    expect(b.needsForceClose()).toBe(false)
    expect(b.remaining()).toBe(5)
  })

  it('startInThinking is ignored when thinkTokens is undefined', () => {
    const b = new ThinkingBudget(undefined, 100, 5, true)
    expect(b.thinking).toBe(false)
  })

  it('exposes a default THINKING_TOKEN_BUDGET of 2048', () => {
    expect(THINKING_TOKEN_BUDGET).toBe(2048)
  })
})
