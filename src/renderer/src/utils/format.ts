/**
 * Strip Electron's IPC wrapper prefix from a thrown value so the user sees only
 * the actual error message. Electron's `ipcRenderer.invoke` reformats main-side
 * errors as `"Error invoking remote method '<channel>': Error: <real msg>"`
 * (the inner `Error:` is sometimes absent). Returns the original string for
 * non-IPC errors.
 */
export function friendlyIpcError(e: unknown): string {
  const raw = e instanceof Error ? e.message : String(e)
  const m = /^Error invoking remote method '[^']+':\s*(?:Error:\s*)?(.+)$/s.exec(raw)
  return m ? m[1] : raw
}

/** Collapse an array of block indices into compact range notation (e.g. "1-3, 5, 7-8"). */
export function formatMissingRanges(missing: number[]): string {
  if (missing.length === 0) return 'none'
  const ranges: string[] = []
  let start = missing[0]
  let end = missing[0]
  for (let i = 1; i < missing.length; i++) {
    if (missing[i] === end + 1) {
      end = missing[i]
    } else {
      ranges.push(start === end ? `${start}` : `${start}-${end}`)
      start = missing[i]
      end = missing[i]
    }
  }
  ranges.push(start === end ? `${start}` : `${start}-${end}`)
  return ranges.join(', ')
}
