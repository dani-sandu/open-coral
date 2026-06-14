import { describe, it, expect } from 'bun:test'
import { friendlyIpcError, formatMissingRanges } from '../../src/renderer/src/utils/format'

describe('friendlyIpcError', () => {
  it("strips Electron's `Error invoking remote method '...': Error: ` wrapper", () => {
    const e = new Error(
      "Error invoking remote method 'opencoral:send-turn': Error: No host available — start hosting or wait for a peer.",
    )
    expect(friendlyIpcError(e)).toBe('No host available — start hosting or wait for a peer.')
  })

  it("strips the wrapper when the inner `Error:` is absent", () => {
    const e = new Error("Error invoking remote method 'opencoral:check-coverage': something went wrong")
    expect(friendlyIpcError(e)).toBe('something went wrong')
  })

  it('returns non-IPC error messages unchanged', () => {
    expect(friendlyIpcError(new Error('plain error'))).toBe('plain error')
  })

  it('coerces non-Error throws to string', () => {
    expect(friendlyIpcError('raw string thrown')).toBe('raw string thrown')
    expect(friendlyIpcError(42)).toBe('42')
  })

  it('handles multi-line inner messages', () => {
    const e = new Error(
      "Error invoking remote method 'x': Error: line one\nline two",
    )
    expect(friendlyIpcError(e)).toBe('line one\nline two')
  })
})

describe('formatMissingRanges', () => {
  it('returns "none" for an empty list', () => {
    expect(formatMissingRanges([])).toBe('none')
  })

  it('collapses runs into ranges', () => {
    expect(formatMissingRanges([1, 2, 3, 5, 7, 8])).toBe('1-3, 5, 7-8')
  })
})
