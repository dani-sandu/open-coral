#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>

// Include real ggml types
#include "ggml.h"

struct ModelConfig {
    int32_t n_embd         = 0;
    int32_t n_head         = 0;
    int32_t n_kv_head      = 0;
    int32_t n_ff           = 0;
    int32_t n_vocab        = 0;
    float   rms_norm_eps   = 1e-5f;
    float   rope_freq_base = 10000.0f;

    // Architecture-specific flags
    float   embed_scale       = 1.0f;   // Gemma: sqrt(n_embd)
    bool    norm_weight_plus1 = false;   // Gemma: RMS norm uses (1 + w) * x
    float   attn_logit_cap    = 0.0f;   // Gemma-2: soft-cap attention logits (0 = off)
    float   final_logit_cap   = 0.0f;   // Gemma-2: soft-cap final logits (0 = off)

    // Gemma-2 post-norms (applied after attention and FFN, before residual)
    bool    use_post_attn_norm = false;
    bool    use_post_ffn_norm  = false;
};

struct LayerWeights {
    ggml_tensor* attn_norm  = nullptr;
    ggml_tensor* attn_q     = nullptr;
    ggml_tensor* attn_k     = nullptr;
    ggml_tensor* attn_v     = nullptr;
    ggml_tensor* attn_o     = nullptr;
    ggml_tensor* ffn_norm   = nullptr;
    ggml_tensor* ffn_gate   = nullptr;
    ggml_tensor* ffn_up     = nullptr;
    ggml_tensor* ffn_down   = nullptr;
    // Gemma-2 post-norms
    ggml_tensor* post_attn_norm = nullptr;
    ggml_tensor* post_ffn_norm  = nullptr;
};

// KV cache for a single inference session
struct SessionState {
    int max_length     = 0;   // allocated sequence capacity
    int prefix_length  = 0;   // number of tokens already cached
    int head_dim       = 0;
    int n_kv_head      = 0;
    int n_layers       = 0;
    // Per-layer KV cache, layout [head_dim * n_kv_head * max_length]
    std::vector<std::vector<float>> k_cache;
    std::vector<std::vector<float>> v_cache;
};

struct ModelContext {
    ggml_context*          weight_ctx  = nullptr;  // tensor metadata context (from gguf_init)
    std::vector<uint8_t>   file_buf;               // raw GGUF file — tensors point into here
    ModelConfig            config;
    int                    block_start = 0;
    int                    block_end   = 0;
    ggml_tensor*           embd_weight = nullptr;
    ggml_tensor*           output_norm = nullptr;
    ggml_tensor*           output_wt   = nullptr;
    std::vector<LayerWeights> layers;

    // Session management for KV caching
    std::unordered_map<int, SessionState*> sessions;
    int next_session_id = 1;
};

ModelContext* model_context_load(const char* path, int block_start, int block_end, int total_blocks);
void          model_context_free(ModelContext* mc);
