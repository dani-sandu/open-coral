import { GGUFHeader, GGUFTensorInfo, BlockRange, ExtractedBlocks } from './types'

const BLOCK_TENSOR_RE = /^blk\.(\d+)\./
const EMBEDDING_TENSORS = new Set(['token_embd.weight'])
const OUTPUT_TENSOR_RE = /^output/

/**
 * Count the number of transformer blocks in the model.
 * Prefers the authoritative metadata key ({arch}.block_count) which is correct
 * even for partial/sharded GGUF files. Falls back to scanning tensor names.
 */
export function countBlocks(header: GGUFHeader): number {
  const arch = (header.metadata.find(m => m.key === 'general.architecture')?.value as string) ?? 'llama'
  const fromMeta = header.metadata.find(m => m.key === `${arch}.block_count`)?.value as number | undefined
  if (fromMeta !== undefined) return Number(fromMeta)

  let max = -1
  for (const tensor of header.tensors) {
    const m = BLOCK_TENSOR_RE.exec(tensor.name)
    if (m) max = Math.max(max, parseInt(m[1], 10))
  }
  return max + 1
}

/**
 * Filter tensors from a parsed GGUF header for the given block range.
 *
 * @param header       Parsed GGUF header from parseGGUFHeader()
 * @param range        Block range [start..end] (inclusive)
 * @param totalBlocks  Total blocks in the model (auto-detected if omitted)
 */
export function extractBlockTensors(
  header: GGUFHeader,
  range: BlockRange,
  totalBlocks?: number
): ExtractedBlocks {
  const total = totalBlocks ?? countBlocks(header)
  if (range.end >= total) {
    throw new Error(`Block range [${range.start}..${range.end}] exceeds model block count (${total})`)
  }

  const blockTensors: GGUFTensorInfo[] = []
  let embeddingTensor: GGUFTensorInfo | null = null
  const outputTensors: GGUFTensorInfo[] = []

  for (const tensor of header.tensors) {
    const blockMatch = BLOCK_TENSOR_RE.exec(tensor.name)

    if (blockMatch) {
      const blockIdx = parseInt(blockMatch[1], 10)
      if (blockIdx >= range.start && blockIdx <= range.end) {
        blockTensors.push(tensor)
      }
      continue
    }

    if (range.start === 0 && EMBEDDING_TENSORS.has(tensor.name)) {
      embeddingTensor = tensor
      continue
    }

    if (totalBlocks !== undefined && range.end === total - 1 && OUTPUT_TENSOR_RE.test(tensor.name)) {
      outputTensors.push(tensor)
    }

    // Weight tying: if hosting the last block but not block 0, include the
    // embedding weight so it can be used as the output projection fallback.
    if (
      totalBlocks !== undefined &&
      range.end === total - 1 &&
      range.start !== 0 &&
      EMBEDDING_TENSORS.has(tensor.name)
    ) {
      outputTensors.push(tensor)
    }
  }

  return { range, blockTensors, embeddingTensor, outputTensors }
}
