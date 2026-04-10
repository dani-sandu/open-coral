#include "block_forward.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

// ── Single block ──────────────────────────────────────────────────────────────
// cctx: scratch context for this computation
// inpL: [n_embd, n_tokens] float32 — input hidden states
// pos:  [n_tokens] int32 — position indices for RoPE
// Returns: [n_embd, n_tokens] float32 — output hidden states

static ggml_tensor* build_block(
    ggml_context*       cctx,
    const LayerWeights& lw,
    const ModelConfig&  cfg,
    ggml_tensor*        inpL,
    ggml_tensor*        pos,
    int                 n_tokens,
    int                 // block_idx — reserved for future KV-cache position offset; positions always 0..n_tokens-1
) {
    const int head_dim = cfg.n_embd / cfg.n_head;
    const float scale  = 1.0f / sqrtf((float)head_dim);

    // ── 1. Attention layer norm ──────────────────────────────────────────────
    ggml_tensor* attn_norm_in = ggml_rms_norm(cctx, inpL, cfg.rms_norm_eps);
    ggml_tensor* cur          = ggml_mul(cctx, attn_norm_in, lw.attn_norm);

    // ── 2. Q / K / V projections ─────────────────────────────────────────────
    ggml_tensor* Qcur = ggml_mul_mat(cctx, lw.attn_q, cur);
    ggml_tensor* Kcur = ggml_mul_mat(cctx, lw.attn_k, cur);
    ggml_tensor* Vcur = ggml_mul_mat(cctx, lw.attn_v, cur);

    // Reshape to [head_dim, n_heads, n_tokens]
    Qcur = ggml_reshape_3d(cctx, Qcur, head_dim, cfg.n_head,    n_tokens);
    Kcur = ggml_reshape_3d(cctx, Kcur, head_dim, cfg.n_kv_head, n_tokens);
    Vcur = ggml_reshape_3d(cctx, Vcur, head_dim, cfg.n_kv_head, n_tokens);

    // ── 3. RoPE ──────────────────────────────────────────────────────────────
    // b = position tensor [n_tokens] i32, c = freq factors (nullptr)
    Qcur = ggml_rope_ext(cctx, Qcur, pos, nullptr,
        head_dim, 0, 0, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    Kcur = ggml_rope_ext(cctx, Kcur, pos, nullptr,
        head_dim, 0, 0, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // ── 4. Scaled dot-product attention ──────────────────────────────────────
    // Q,K,V are [head_dim, n_heads, n_tokens] — permute to [head_dim, n_tokens, n_heads]
    ggml_tensor* Q_p = ggml_permute(cctx, Qcur, 0, 2, 1, 3);
    ggml_tensor* K_p = ggml_permute(cctx, Kcur, 0, 2, 1, 3);
    ggml_tensor* V_p = ggml_permute(cctx, Vcur, 0, 2, 1, 3);

    // KQ = K^T @ Q -> [n_tokens, n_tokens, n_head]
    ggml_tensor* KQ = ggml_mul_mat(cctx, K_p, Q_p);
    KQ = ggml_scale(cctx, KQ, scale);
    KQ = ggml_soft_max(cctx, KQ);

    // Transpose V: [head_dim, n_tokens, n_kv_head] -> [n_tokens, head_dim, n_kv_head]
    ggml_tensor* V_t = ggml_cont(cctx, ggml_transpose(cctx, V_p));

    // KQV = V_t^T @ KQ -> [head_dim, n_tokens, n_head]
    ggml_tensor* KQV = ggml_mul_mat(cctx, V_t, KQ);
    KQV = ggml_permute(cctx, KQV, 0, 2, 1, 3);
    KQV = ggml_cont(cctx, KQV);
    KQV = ggml_reshape_2d(cctx, KQV, cfg.n_embd, n_tokens);

    // ── 5. Output projection + residual ──────────────────────────────────────
    ggml_tensor* attn_out = ggml_mul_mat(cctx, lw.attn_o, KQV);
    ggml_tensor* inpFF    = ggml_add(cctx, attn_out, inpL);

    // ── 6. FFN layer norm ─────────────────────────────────────────────────────
    ggml_tensor* ffn_norm_in = ggml_rms_norm(cctx, inpFF, cfg.rms_norm_eps);
    ggml_tensor* ff_cur      = ggml_mul(cctx, ffn_norm_in, lw.ffn_norm);

    // ── 7. SwiGLU FFN ────────────────────────────────────────────────────────
    ggml_tensor* gate    = ggml_mul_mat(cctx, lw.ffn_gate, ff_cur);
    ggml_tensor* up      = ggml_mul_mat(cctx, lw.ffn_up,   ff_cur);
    gate                 = ggml_silu(cctx, gate);
    ggml_tensor* ffn_act = ggml_mul(cctx, gate, up);
    ggml_tensor* ffn_out = ggml_mul_mat(cctx, lw.ffn_down, ffn_act);

    // ── 8. Residual ───────────────────────────────────────────────────────────
    return ggml_add(cctx, ffn_out, inpFF);
}

// ── Public ────────────────────────────────────────────────────────────────────

std::vector<float> block_runner_forward(
    ModelContext* mc,
    const float*  input_data,
    int           n_tokens
) {
    const ModelConfig& cfg    = mc->config;
    const int          n_embd = cfg.n_embd;
    const int          n_out  = n_tokens * n_embd;

    // 256 MB scratch context for computation graph.
    // NOTE: size is not validated against actual n_tokens/n_embd requirements.
    // Individual tensor allocations within the context are not checked for nullptr.
    // For real models (n_tokens > 512 or n_embd > 4096), increase as needed.
    const size_t scratch_mb = 256;
    std::vector<uint8_t> scratch(scratch_mb * 1024 * 1024);

    ggml_init_params cparams{
        scratch.size(),
        scratch.data(),
        false
    };
    ggml_context* cctx = ggml_init(cparams);
    if (!cctx) throw std::runtime_error("block_runner_forward: ggml_init failed");

    // Input tensor [n_embd, n_tokens]
    ggml_tensor* inpL = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(inpL, "input_hidden");
    memcpy(inpL->data, input_data, (size_t)n_out * sizeof(float));

    // Position tensor [n_tokens] i32 — sequential positions 0..n_tokens-1
    ggml_tensor* pos = ggml_new_tensor_1d(cctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(pos, "positions");
    int32_t* pos_data = (int32_t*)pos->data;
    for (int i = 0; i < n_tokens; i++) {
        pos_data[i] = i;
    }

    // Chain blocks
    ggml_tensor* cur = inpL;
    const int n_layers = mc->block_end - mc->block_start + 1;
    for (int i = 0; i < n_layers; i++) {
        cur = build_block(cctx, mc->layers[i], cfg, cur, pos, n_tokens, mc->block_start + i);
    }
    ggml_set_name(cur, "output_hidden");

    // Build and execute graph
    ggml_cgraph* graph = ggml_new_graph(cctx);
    ggml_build_forward_expand(graph, cur);

    int n_threads = 4;
    ggml_status status = ggml_graph_compute_with_ctx(cctx, graph, n_threads);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(cctx);
        throw std::runtime_error("ggml_graph_compute_with_ctx failed");
    }

    // Copy output
    std::vector<float> result((size_t)n_out);
    memcpy(result.data(), cur->data, (size_t)n_out * sizeof(float));

    ggml_free(cctx);
    return result;
}
