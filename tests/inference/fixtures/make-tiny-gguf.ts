import { writeFileSync } from 'fs'

/** Small fixed dimensions that make tests fast */
export const TINY_CONFIG = {
  n_embd:     16,
  n_head:     2,
  n_kv_head:  2,
  n_ff:       32,
  n_blocks:   2,
  vocab_size: 64,
  rms_norm_eps:   1e-5,
  rope_freq_base: 10000.0,
} as const

// ── Binary writer ──────────────────────────────────────────────────────────
class BufWriter {
  parts: Buffer[] = []
  u32(v: number) { const b = Buffer.allocUnsafe(4); b.writeUInt32LE(v); this.parts.push(b) }
  u64(v: bigint) { const b = Buffer.allocUnsafe(8); b.writeBigUInt64LE(v); this.parts.push(b) }
  f32(v: number) { const b = Buffer.allocUnsafe(4); b.writeFloatLE(v);  this.parts.push(b) }
  str(s: string) { this.u64(BigInt(s.length)); this.parts.push(Buffer.from(s, 'utf8')) }
  /** Write a metadata KV with UINT32 value (type = 4) */
  kvU32(key: string, val: number) { this.str(key); this.u32(4); this.u32(val) }
  /** Write a metadata KV with FLOAT32 value (type = 6) */
  kvF32(key: string, val: number) { this.str(key); this.u32(6); this.f32(val) }
  /** Write a metadata KV with STRING value (type = 8) */
  kvStr(key: string, val: string) { this.str(key); this.u32(8); this.str(val) }
  toBuffer() { return Buffer.concat(this.parts) }
}

// ── Tensor registry ────────────────────────────────────────────────────────
interface TensorDef { name: string; dims: number[]; data: Float32Array }

function addTensor(list: TensorDef[], name: string, ...dims: number[]) {
  const size = dims.reduce((a, b) => a * b, 1)
  // Use small non-zero values to avoid degenerate outputs in forward pass tests
  const data = new Float32Array(size).fill(0.01)
  list.push({ name, dims, data })
}

/**
 * Build a minimal valid GGUF v3 binary for a 2-block LLaMA-style model.
 * All weights are F32, filled with 0.01.
 * File is small (~35 KB) so tests are fast.
 *
 * Dimension convention: dims are written in ggml column-major order where
 * ne[0] = in_features (innermost / fastest-varying dimension). For weight
 * matrices this means (in_features, out_features), e.g. attn_q is (n_embd, n_embd).
 * This matches the layout that llama.cpp expects when loading GGUF tensors.
 */
export function buildTinyGGUF(): Buffer {
  const cfg = TINY_CONFIG
  const head_dim = cfg.n_embd / cfg.n_head
  const kv_dim   = cfg.n_kv_head * head_dim

  const tensors: TensorDef[] = []

  addTensor(tensors, 'token_embd.weight', cfg.n_embd, cfg.vocab_size)

  for (let i = 0; i < cfg.n_blocks; i++) {
    addTensor(tensors, `blk.${i}.attn_norm.weight`,   cfg.n_embd)
    addTensor(tensors, `blk.${i}.attn_q.weight`,      cfg.n_embd, cfg.n_embd)
    addTensor(tensors, `blk.${i}.attn_k.weight`,      cfg.n_embd, kv_dim)
    addTensor(tensors, `blk.${i}.attn_v.weight`,      cfg.n_embd, kv_dim)
    addTensor(tensors, `blk.${i}.attn_output.weight`, cfg.n_embd, cfg.n_embd)
    addTensor(tensors, `blk.${i}.ffn_norm.weight`,    cfg.n_embd)
    addTensor(tensors, `blk.${i}.ffn_gate.weight`,    cfg.n_embd, cfg.n_ff)
    addTensor(tensors, `blk.${i}.ffn_up.weight`,      cfg.n_embd, cfg.n_ff)
    addTensor(tensors, `blk.${i}.ffn_down.weight`,    cfg.n_ff,   cfg.n_embd)
  }

  addTensor(tensors, 'output_norm.weight', cfg.n_embd)
  addTensor(tensors, 'output.weight',      cfg.n_embd, cfg.vocab_size)

  // ── Header ──────────────────────────────────────────────────────────────
  const w = new BufWriter()

  w.parts.push(Buffer.from('GGUF', 'ascii'))
  w.u32(3)                          // version = 3
  w.u64(BigInt(tensors.length))     // tensor_count
  w.u64(11n)                        // metadata_kv_count = 11

  // Metadata
  w.kvStr('general.architecture',                  'llama')
  w.kvU32('llama.context_length',                  2048)
  w.kvU32('llama.block_count',                     cfg.n_blocks)
  w.kvU32('llama.vocab_size',                      cfg.vocab_size)
  w.kvU32('llama.embedding_length',                cfg.n_embd)
  w.kvU32('llama.attention.head_count',            cfg.n_head)
  w.kvU32('llama.attention.head_count_kv',         cfg.n_kv_head)
  w.kvU32('llama.feed_forward_length',             cfg.n_ff)
  w.kvF32('llama.rope.freq_base',                  cfg.rope_freq_base)
  w.kvF32('llama.attention.layer_norm_rms_epsilon', cfg.rms_norm_eps)
  // "no_vocab" tells llama.cpp to skip tokenizer loading (valid for test fixtures)
  w.kvStr('tokenizer.ggml.model',                  'no_vocab')

  // Tensor info
  let offset = 0n
  for (const t of tensors) {
    w.str(t.name)
    w.u32(t.dims.length)
    for (const d of t.dims) w.u64(BigInt(d))
    w.u32(0)              // F32 type
    w.u64(offset)
    offset += BigInt(t.data.byteLength)
  }

  // Align to 32 bytes
  const header = w.toBuffer()
  const padLen  = (32 - (header.length % 32)) % 32
  const padded  = Buffer.concat([header, Buffer.alloc(padLen)])

  // Tensor data
  const dataBufs = tensors.map(t => Buffer.from(t.data.buffer))
  return Buffer.concat([padded, ...dataBufs])
}

// CLI: write to file when run directly
if (process.argv[2]) {
  const buf = buildTinyGGUF()
  writeFileSync(process.argv[2], buf)
  console.log(`Written tiny GGUF (${buf.length} bytes) to ${process.argv[2]}`)
}
