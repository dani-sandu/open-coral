#pragma once
#include <cstdint>
#include <vector>
#include <unordered_map>
#include "llama.h"

struct SessionInfo {
    int seq_id;
    int n_past;
    int max_length;
};

struct LlamaBlockContext {
    llama_model*   model       = nullptr;
    llama_context* ctx         = nullptr;
    int32_t block_start        = 0;
    int32_t block_end          = 0;
    int32_t total_blocks       = 0;
    int     next_session_id    = 1;
    int     next_seq_id        = 1; // 0 is reserved for stateless calls
    std::unordered_map<int, SessionInfo> sessions;
    std::vector<float> shim_logits; // last-token logits cached by lbc_embed_tokens in shim mode
};

// block_start and block_end are both INCLUSIVE (e.g., blocks 0..27 for a 28-block model).
// block_end == -1 is a special shim-mode sentinel: no transformer blocks are run, but
// embedTokens() and projectToLogits() are available (full model weights are loaded).
// seq 0 is reserved internally for stateless calls; session seq_ids start at 1.

LlamaBlockContext* lbc_load(
    const char* path,
    int32_t block_start, int32_t block_end, int32_t total_blocks);

LlamaBlockContext* lbc_load_sharded(
    const char** paths, int n_shards,
    int32_t block_start, int32_t block_end, int32_t total_blocks);

void lbc_free(LlamaBlockContext* lbc);

// First-node only (block_start == 0): token IDs -> hidden states after embedding lookup.
std::vector<float> lbc_embed_tokens(
    LlamaBlockContext* lbc, const int32_t* ids, int n_tokens);

// Last-node only (block_end == total_blocks - 1): hidden -> logits for the last token only.
// Returns n_vocab floats regardless of n_tokens.
std::vector<float> lbc_project_to_logits(
    LlamaBlockContext* lbc, const float* hidden, int n_tokens);

// Stateless block-range forward pass: hidden_in -> hidden_out. Clears seq 0 KV before each call.
std::vector<float> lbc_forward(
    LlamaBlockContext* lbc, const float* hidden_in, int n_tokens);

// KV-cached decode session
int  lbc_session_open(LlamaBlockContext* lbc, int max_seq_len);
void lbc_session_close(LlamaBlockContext* lbc, int session_id);
std::vector<float> lbc_session_forward(
    LlamaBlockContext* lbc, int session_id,
    const float* hidden_in, int n_new_tokens);

// Full-model KV-cached decode: token IDs -> logits for the last token.
// Uses the session's KV cache slot so context accumulates across calls.
// Only valid on shim contexts (block_end == -1) that load the full model.
std::vector<float> lbc_session_decode_logits(
    LlamaBlockContext* lbc, int session_id,
    const int32_t* ids, int n_tokens);
