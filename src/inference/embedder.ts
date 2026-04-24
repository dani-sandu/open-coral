/**
 * Converts token IDs to hidden states.
 *
 * Invariants (required by KVChain):
 * - Stateless: calling embed() must not mutate any session's KV cache.
 * - Position-independent: output depends only on token IDs, not on where
 *   they will be placed in the sequence. This holds for architectures
 *   that apply positional encoding inside attention (e.g., RoPE in LLaMA).
 *   It does NOT hold for architectures that add position at the embedding
 *   stage (classic BERT, GPT-2 with learned position embeddings).
 */
export interface Embedder {
  readonly nEmbd: number
  embed(tokenIds: Int32Array): Promise<Float32Array>
}
