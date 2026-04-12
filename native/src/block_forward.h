#pragma once
#include "model_context.h"
#include <vector>

// Run hidden states through loaded transformer blocks.
// input_data: float32 values, length = n_tokens × config.n_embd
// Returns float32 output, same length.
std::vector<float> block_runner_forward(
    ModelContext* mc,
    const float*  input_data,
    int           n_tokens
);

// Look up token embeddings from the embedding weight matrix.
// Requires mc->embd_weight (block_start == 0).
// token_ids: int32 values, length = n_tokens
// Returns float32 output, length = n_tokens × config.n_embd.
std::vector<float> embed_tokens(
    ModelContext*  mc,
    const int32_t* token_ids,
    int            n_tokens
);

// Apply final RMS norm and project hidden states to vocabulary logits.
// Requires mc->output_norm and mc->output_wt (block_end == last block).
// hidden_states: float32 values, length = n_tokens × config.n_embd
// Returns float32 logits, length = n_tokens × config.n_vocab.
std::vector<float> project_to_logits(
    ModelContext* mc,
    const float*  hidden_states,
    int           n_tokens
);

// ── KV-cached inference sessions ──────────────────────────────────────────────

// Allocate a KV cache for max_length tokens across all hosted blocks.
// Returns a session ID for use with session_forward / free_session.
int  create_session(ModelContext* mc, int max_length);

// Free a session's KV cache.
void free_session(ModelContext* mc, int session_id);

// Forward pass with KV caching.  Only the n_new_tokens are processed;
// previously cached K/V entries are reused for attention.
// input_data: float32, length = n_new_tokens × config.n_embd
// Returns float32 output, same length.
std::vector<float> session_forward(
    ModelContext* mc,
    int           session_id,
    const float*  input_data,
    int           n_new_tokens
);
