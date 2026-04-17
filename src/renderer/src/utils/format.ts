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
