import { GGUFHeader, GGUFMetadataEntry, GGUFTensorInfo, GGUFValueType, GGMLType } from './types'

class GGUFReader {
  private offset = 0
  constructor(private readonly buf: Buffer) {}

  readBytes(n: number): Buffer {
    if (this.offset + n > this.buf.length) {
      throw new Error(`GGUFReader: read of ${n} bytes at offset ${this.offset} exceeds buffer length ${this.buf.length}`)
    }
    const slice = this.buf.subarray(this.offset, this.offset + n)
    this.offset += n
    return slice
  }
  readUint8(): number {
    const v = this.buf.readUInt8(this.offset); this.offset += 1; return v
  }
  readInt8(): number {
    const v = this.buf.readInt8(this.offset); this.offset += 1; return v
  }
  readUint16(): number {
    const v = this.buf.readUInt16LE(this.offset); this.offset += 2; return v
  }
  readInt16(): number {
    const v = this.buf.readInt16LE(this.offset); this.offset += 2; return v
  }
  readUint32(): number {
    const v = this.buf.readUInt32LE(this.offset); this.offset += 4; return v
  }
  readInt32(): number {
    const v = this.buf.readInt32LE(this.offset); this.offset += 4; return v
  }
  readUint64(): bigint {
    const v = this.buf.readBigUInt64LE(this.offset); this.offset += 8; return v
  }
  readInt64(): bigint {
    const v = this.buf.readBigInt64LE(this.offset); this.offset += 8; return v
  }
  readFloat32(): number {
    const v = this.buf.readFloatLE(this.offset); this.offset += 4; return v
  }
  readFloat64(): number {
    const v = this.buf.readDoubleLE(this.offset); this.offset += 8; return v
  }
  readBool(): boolean {
    return this.readUint8() !== 0
  }
  readString(): string {
    const len = Number(this.readUint64())
    return this.readBytes(len).toString('utf8')
  }
  get position(): number { return this.offset }
  alignTo(n: number): void {
    const rem = this.offset % n
    if (rem !== 0) this.offset += n - rem
  }
}

function readMetadataValue(reader: GGUFReader, valueType: GGUFValueType): unknown {
  switch (valueType) {
    case GGUFValueType.UINT8:   return reader.readUint8()
    case GGUFValueType.INT8:    return reader.readInt8()
    case GGUFValueType.UINT16:  return reader.readUint16()
    case GGUFValueType.INT16:   return reader.readInt16()
    case GGUFValueType.UINT32:  return reader.readUint32()
    case GGUFValueType.INT32:   return reader.readInt32()
    case GGUFValueType.FLOAT32: return reader.readFloat32()
    case GGUFValueType.BOOL:    return reader.readBool()
    case GGUFValueType.STRING:  return reader.readString()
    case GGUFValueType.UINT64:  return reader.readUint64()
    case GGUFValueType.INT64:   return reader.readInt64()
    case GGUFValueType.FLOAT64: return reader.readFloat64()
    case GGUFValueType.ARRAY: {
      const elemType = reader.readUint32() as GGUFValueType
      const count = Number(reader.readUint64())
      return Array.from({ length: count }, () => readMetadataValue(reader, elemType))
    }
    default:
      throw new Error(`Unknown GGUF value type: ${valueType}`)
  }
}

/**
 * Parse the header of a GGUF file buffer.
 * Does NOT load tensor data — only reads metadata and tensor layout info.
 */
export function parseGGUFHeader(buf: Buffer): GGUFHeader {
  const reader = new GGUFReader(buf)

  // Magic
  const magic = reader.readBytes(4).toString('ascii')
  if (magic !== 'GGUF') throw new Error(`Invalid GGUF magic: expected "GGUF", got "${magic}"`)

  // Version
  const version = reader.readUint32()
  if (version < 2 || version > 3) throw new Error(`Unsupported GGUF version: ${version}`)

  const tensorCount = reader.readUint64()
  const metadataKvCount = reader.readUint64()

  const MAX_SAFE_COUNT = 1_000_000n
  if (tensorCount > MAX_SAFE_COUNT || metadataKvCount > MAX_SAFE_COUNT) {
    throw new Error(`GGUF header declares implausible count (tensors: ${tensorCount}, kv: ${metadataKvCount})`)
  }

  // Metadata
  const metadata: GGUFMetadataEntry[] = []
  for (let i = 0; i < Number(metadataKvCount); i++) {
    const key = reader.readString()
    const valueType = reader.readUint32() as GGUFValueType
    const value = readMetadataValue(reader, valueType)
    metadata.push({ key, valueType, value })
  }

  // Tensor info
  const tensors: GGUFTensorInfo[] = []
  for (let i = 0; i < Number(tensorCount); i++) {
    const name = reader.readString()
    const nDims = reader.readUint32()
    const shape: bigint[] = []
    for (let d = 0; d < nDims; d++) shape.push(reader.readUint64())
    const type = reader.readUint32() as GGMLType
    const dataOffset = reader.readUint64()
    tensors.push({ name, shape, type, dataOffset })
  }

  // Align to 32 bytes — this is where the data region starts
  reader.alignTo(32)
  const dataRegionOffset = BigInt(reader.position)

  return { version, tensorCount, metadata, tensors, dataRegionOffset }
}
