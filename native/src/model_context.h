#pragma once
#include <cstdint>
#include <vector>
#include <string>

// Include real ggml types
#include "ggml.h"

struct ModelConfig {
    int32_t n_embd         = 0;
    int32_t n_head         = 0;
    int32_t n_kv_head      = 0;
    int32_t n_ff           = 0;
    float   rms_norm_eps   = 1e-5f;
    float   rope_freq_base = 10000.0f;
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
};

ModelContext* model_context_load(const char* path, int block_start, int block_end, int total_blocks);
void          model_context_free(ModelContext* mc);
