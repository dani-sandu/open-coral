export function sampleTopK(
  logits: Float32Array,
  vocabSize: number,
  offset: number,
  temperature = 0.7,
  topK = 40,
): number {
  const candidates: { idx: number; logit: number }[] = []
  for (let i = 0; i < vocabSize; i++) {
    candidates.push({ idx: i, logit: logits[offset + i] })
  }
  candidates.sort((a, b) => b.logit - a.logit)
  const top = candidates.slice(0, topK)
  const scaled = top.map(c => c.logit / temperature)
  const maxVal = scaled[0]
  const exps = scaled.map(v => Math.exp(v - maxVal))
  const sum = exps.reduce((a, b) => a + b, 0)
  const probs = exps.map(e => e / sum)
  const r = Math.random()
  let cumulative = 0
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i]
    if (r <= cumulative) return top[i].idx
  }
  return top[top.length - 1].idx
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
