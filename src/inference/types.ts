/** GGML quantization / tensor types as defined in the GGUF spec */
export enum GGMLType {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K_S = 11,
  Q3_K_M = 12,
  Q3_K_L = 13,
  Q4_K_S = 14,
  Q4_K_M = 15,
  Q5_K_S = 16,
  Q5_K_M = 17,
  Q6_K = 18,
  Q8_K = 19,
  IQ2_XXS = 20,
  IQ2_XS = 21,
  IQ3_XXS = 22,
  IQ1_S = 23,
  IQ4_NL = 24,
  IQ3_S = 25,
  IQ2_S = 26,
  IQ4_XS = 27,
  I8 = 28,
  I16 = 29,
  I32 = 30,
  I64 = 31,
  F64 = 32,
  IQ1_M = 33,
  BF16 = 34,
}

/** GGUF metadata value types */
export enum GGUFValueType {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
}

/** A single tensor's layout info as stored in the GGUF file */
export interface GGUFTensorInfo {
  name: string
  /** Shape dimensions in row-major order */
  shape: bigint[]
  type: GGMLType
  /** Byte offset of this tensor's data from the start of the data region */
  dataOffset: bigint
}

/** Top-level metadata key-value pair from a GGUF file */
export interface GGUFMetadataEntry {
  key: string
  valueType: GGUFValueType
  value: unknown
}

/** Full result of parsing a GGUF file's header */
export interface GGUFHeader {
  version: number
  tensorCount: bigint
  metadata: GGUFMetadataEntry[]
  tensors: GGUFTensorInfo[]
  /** Byte offset where tensor data begins (after header + tensor info) */
  dataRegionOffset: bigint
}

/** A contiguous range of transformer blocks [start, end] (both inclusive) */
export interface BlockRange {
  start: number
  end: number
}

/** Tensors belonging to a block range, plus the special embedding/output tensors */
export interface ExtractedBlocks {
  range: BlockRange
  /** Tensors for the specified block range (blk.{start}..blk.{end}) */
  blockTensors: GGUFTensorInfo[]
  /** Embedding tensor (only present if range.start === 0) */
  embeddingTensor: GGUFTensorInfo | null
  /** LM head tensors (only present if range.end is the last block) */
  outputTensors: GGUFTensorInfo[]
}
