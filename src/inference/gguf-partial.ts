import { GGUFHeader, GGUFTensorInfo, GGMLType } from './types'

/**
 * GGML type → (block size in elements, byte size per block)
 * Used to calculate how many bytes a tensor's data occupies.
 * Values must match ggml.h enum and struct sizes exactly.
 */
const GGML_TYPE_META: Record<number, { blockSize: number; typeSize: number }> = {
  [GGMLType.F32]:     { blockSize: 1, typeSize: 4 },
  [GGMLType.F16]:     { blockSize: 1, typeSize: 2 },
  [GGMLType.Q4_0]:    { blockSize: 32, typeSize: 18 },
  [GGMLType.Q4_1]:    { blockSize: 32, typeSize: 20 },
  [GGMLType.Q5_0]:    { blockSize: 32, typeSize: 22 },
  [GGMLType.Q5_1]:    { blockSize: 32, typeSize: 24 },
  [GGMLType.Q8_0]:    { blockSize: 32, typeSize: 34 },
  [GGMLType.Q8_1]:    { blockSize: 32, typeSize: 36 },
  [GGMLType.Q2_K]:    { blockSize: 256, typeSize: 84 },
  [GGMLType.Q3_K]:    { blockSize: 256, typeSize: 110 },
  [GGMLType.Q4_K]:    { blockSize: 256, typeSize: 144 },
  [GGMLType.Q5_K]:    { blockSize: 256, typeSize: 176 },
  [GGMLType.Q6_K]:    { blockSize: 256, typeSize: 210 },
  [GGMLType.Q8_K]:    { blockSize: 256, typeSize: 292 },
  [GGMLType.IQ2_XXS]: { blockSize: 256, typeSize: 66 },
  [GGMLType.IQ2_XS]:  { blockSize: 256, typeSize: 74 },
  [GGMLType.IQ3_XXS]: { blockSize: 256, typeSize: 98 },
  [GGMLType.IQ1_S]:   { blockSize: 256, typeSize: 50 },
  [GGMLType.IQ4_NL]:  { blockSize: 32, typeSize: 18 },
  [GGMLType.IQ3_S]:   { blockSize: 256, typeSize: 110 },
  [GGMLType.IQ2_S]:   { blockSize: 256, typeSize: 82 },
  [GGMLType.IQ4_XS]:  { blockSize: 256, typeSize: 136 },
  [GGMLType.I8]:      { blockSize: 1, typeSize: 1 },
  [GGMLType.I16]:     { blockSize: 1, typeSize: 2 },
  [GGMLType.I32]:     { blockSize: 1, typeSize: 4 },
  [GGMLType.I64]:     { blockSize: 1, typeSize: 8 },
  [GGMLType.F64]:     { blockSize: 1, typeSize: 8 },
  [GGMLType.IQ1_M]:   { blockSize: 256, typeSize: 56 },
  [GGMLType.BF16]:    { blockSize: 1, typeSize: 2 },
}

/** Compute the byte size of a tensor's data region from shape and type. */
export function tensorDataSize(shape: bigint[], type: GGMLType): number {
  const nElements = shape.reduce((a, b) => a * b, 1n)
  const meta = GGML_TYPE_META[type]
  if (!meta) throw new Error(`Unknown GGML type: ${type}`)
  const nBlocks = Number(nElements) / meta.blockSize
  return Math.ceil(nBlocks) * meta.typeSize
}

/** Absolute byte range [offset, offset+size) in the original file for a tensor. */
export interface TensorByteRange {
  name: string
  absoluteOffset: number   // dataRegionOffset + tensor.dataOffset
  size: number             // computed from shape + type
}

/** Calculate absolute byte ranges for a list of tensors within a GGUF file. */
export function tensorByteRanges(
  header: GGUFHeader,
  tensors: GGUFTensorInfo[],
): TensorByteRange[] {
  const dataStart = Number(header.dataRegionOffset)
  return tensors.map(t => ({
    name: t.name,
    absoluteOffset: dataStart + Number(t.dataOffset),
    size: tensorDataSize(t.shape, t.type),
  }))
}

const GGUF_ALIGNMENT = 32

function align(n: number, alignment: number): number {
  const rem = n % alignment
  return rem === 0 ? n : n + (alignment - rem)
}

/**
 * Encode a GGUF metadata string: uint64 length + UTF-8 bytes
 */
function encodeGGUFString(s: string): Buffer {
  const strBytes = Buffer.from(s, 'utf8')
  const buf = Buffer.allocUnsafe(8 + strBytes.length)
  buf.writeBigUInt64LE(BigInt(strBytes.length), 0)
  strBytes.copy(buf, 8)
  return buf
}

/**
 * Build a partial GGUF file containing only the specified tensors.
 *
 * @param originalHeaderBuf  The original GGUF file header bytes (first N MB)
 * @param header             Parsed header from parseGGUFHeader()
 * @param selectedTensors    Subset of header.tensors to include
 * @param tensorDataChunks   Map from tensor name → downloaded data buffer
 * @returns                  Complete partial GGUF file as a single Buffer
 */
export function buildPartialGGUF(
  originalHeaderBuf: Buffer,
  header: GGUFHeader,
  selectedTensors: GGUFTensorInfo[],
  tensorDataChunks: Map<string, Buffer>,
): Buffer {
  const parts: Buffer[] = []

  // ── 1. File header: magic + version + n_tensors + n_kv ──────────────────
  const fileHeader = Buffer.allocUnsafe(24)
  fileHeader.write('GGUF', 0, 4, 'ascii')
  fileHeader.writeUInt32LE(header.version, 4)
  fileHeader.writeBigUInt64LE(BigInt(selectedTensors.length), 8)   // reduced tensor count
  fileHeader.writeBigUInt64LE(BigInt(header.metadata.length), 16)
  parts.push(fileHeader)

  // ── 2. Metadata KV pairs — copy raw bytes from original header ──────────
  // Re-serializing is error-prone (tokenizer arrays, nested types, etc.).
  // The metadata region is bytes [24 .. metadataEndOffset) in the original buffer.
  parts.push(originalHeaderBuf.subarray(24, header.metadataEndOffset))

  // ── 3. Tensor info entries (only for selected tensors) ─────────────────
  // We'll pack tensor data contiguously, so compute new offsets.
  let dataOffset = 0n
  for (const t of selectedTensors) {
    // name
    parts.push(encodeGGUFString(t.name))
    // n_dims
    const dimsBuf = Buffer.allocUnsafe(4 + t.shape.length * 8 + 4 + 8)
    let off = 0
    dimsBuf.writeUInt32LE(t.shape.length, off); off += 4
    for (const dim of t.shape) {
      dimsBuf.writeBigUInt64LE(dim, off); off += 8
    }
    // type
    dimsBuf.writeUInt32LE(t.type, off); off += 4
    // data offset (relative to data region start)
    dimsBuf.writeBigUInt64LE(dataOffset, off); off += 8
    parts.push(dimsBuf.subarray(0, off))

    const size = tensorDataSize(t.shape, t.type)
    dataOffset += BigInt(align(size, GGUF_ALIGNMENT))
  }

  // ── 4. Alignment padding to start of data region ───────────────────────
  const headerSize = parts.reduce((sum, b) => sum + b.length, 0)
  const alignedHeaderSize = align(headerSize, GGUF_ALIGNMENT)
  if (alignedHeaderSize > headerSize) {
    parts.push(Buffer.alloc(alignedHeaderSize - headerSize))
  }

  // ── 5. Tensor data — packed contiguously, each aligned ─────────────────
  for (const t of selectedTensors) {
    const data = tensorDataChunks.get(t.name)
    if (!data) throw new Error(`Missing data for tensor: ${t.name}`)
    parts.push(data)
    // Alignment padding
    const aligned = align(data.length, GGUF_ALIGNMENT)
    if (aligned > data.length) {
      parts.push(Buffer.alloc(aligned - data.length))
    }
  }

  return Buffer.concat(parts)
}

/**
 * Estimate the total download size for a partial GGUF with specific tensors.
 */
export function estimatePartialSize(tensors: GGUFTensorInfo[]): number {
  let total = 0
  for (const t of tensors) {
    total += tensorDataSize(t.shape, t.type)
  }
  // Add ~4MB estimate for header
  return total + 4 * 1024 * 1024
}
