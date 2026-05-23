// Reused scratch buffers for the top-K heap. Safe as module-level mutable state
// because sampleTopK is fully synchronous — no two calls ever interleave on them.
// Grown on demand, never shrunk. Indices into logits[offset..], not stable across calls.
let _heapIdx: Uint32Array = new Uint32Array(0)
let _heapLogit: Float32Array = new Float32Array(0)
let _heapExp: Float64Array = new Float64Array(0)

function ensureHeapCapacity(k: number): void {
  if (_heapIdx.length < k) {
    _heapIdx = new Uint32Array(k)
    _heapLogit = new Float32Array(k)
    _heapExp = new Float64Array(k)
  }
}

// Min-heap sift-up / sift-down on _heapLogit, with _heapIdx mirrored in lockstep.
function heapPush(size: number, logit: number, idx: number): void {
  _heapLogit[size] = logit
  _heapIdx[size] = idx
  let i = size
  while (i > 0) {
    const parent = (i - 1) >> 1
    if (_heapLogit[parent] <= _heapLogit[i]) break
    const tl = _heapLogit[parent]; _heapLogit[parent] = _heapLogit[i]; _heapLogit[i] = tl
    const ti = _heapIdx[parent];   _heapIdx[parent]   = _heapIdx[i];   _heapIdx[i]   = ti
    i = parent
  }
}

function heapReplaceMin(size: number, logit: number, idx: number): void {
  _heapLogit[0] = logit
  _heapIdx[0] = idx
  let i = 0
  while (true) {
    const l = 2 * i + 1, r = 2 * i + 2
    let smallest = i
    if (l < size && _heapLogit[l] < _heapLogit[smallest]) smallest = l
    if (r < size && _heapLogit[r] < _heapLogit[smallest]) smallest = r
    if (smallest === i) break
    const tl = _heapLogit[smallest]; _heapLogit[smallest] = _heapLogit[i]; _heapLogit[i] = tl
    const ti = _heapIdx[smallest];   _heapIdx[smallest]   = _heapIdx[i];   _heapIdx[i]   = ti
    i = smallest
  }
}

export function sampleTopK(
  logits: Float32Array,
  vocabSize: number,
  offset: number,
  temperature = 0.7,
  topK = 40,
): number {
  const k = Math.min(topK, vocabSize)
  ensureHeapCapacity(k)

  // Single linear pass building a min-heap of size k — root is the smallest of the top-K.
  // Anything smaller than the root is discarded immediately.
  let heapSize = 0
  for (let i = 0; i < vocabSize; i++) {
    const v = logits[offset + i]
    if (heapSize < k) {
      heapPush(heapSize, v, i)
      heapSize++
    } else if (v > _heapLogit[0]) {
      heapReplaceMin(heapSize, v, i)
    }
  }

  // Softmax over the K survivors with the standard max-subtraction trick.
  // Cache the K exponentials so the cumulative loop does not recompute them.
  let maxLogit = -Infinity
  for (let i = 0; i < heapSize; i++) if (_heapLogit[i] > maxLogit) maxLogit = _heapLogit[i]
  let sum = 0
  for (let i = 0; i < heapSize; i++) {
    const e = Math.exp((_heapLogit[i] - maxLogit) / temperature)
    _heapExp[i] = e
    sum += e
  }
  const r = Math.random() * sum

  let cumulative = 0
  for (let i = 0; i < heapSize; i++) {
    cumulative += _heapExp[i]
    if (r <= cumulative) return _heapIdx[i]
  }
  return _heapIdx[heapSize - 1]
}

/**
 * Returns the softmax probability of `tokenId` given `logits[offset..offset+vocabSize]`.
 * Used in speculative sampling: accept a draft token with probability p_target[draftToken].
 */
export function softmaxProb(
  logits: Float32Array,
  offset: number,
  vocabSize: number,
  tokenId: number,
): number {
  if (tokenId < 0 || tokenId >= vocabSize) {
    throw new RangeError(`tokenId ${tokenId} out of range [0, ${vocabSize})`)
  }
  let maxVal = -Infinity
  for (let i = 0; i < vocabSize; i++) {
    if (logits[offset + i] > maxVal) maxVal = logits[offset + i]
  }
  let sum = 0
  for (let i = 0; i < vocabSize; i++) {
    sum += Math.exp(logits[offset + i] - maxVal)
  }
  return Math.exp(logits[offset + tokenId] - maxVal) / sum
}
