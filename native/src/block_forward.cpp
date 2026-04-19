#include "block_forward.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

// Leave 2 logical cores for the Electron main + renderer processes.
// hardware_concurrency() returns 0 if unknown; fall back to 2.
static const int g_n_threads = [] {
    int hw = (int)std::thread::hardware_concurrency();
    return std::max(1, (hw > 2 ? hw - 2 : 1));
}();

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
    int                 // block_idx — reserved
) {
    // Derive head_dim from the Q weight shape rather than assuming n_embd/n_head.
    // Models like Gemma-2 have head_dim != n_embd/n_head (e.g. 256 vs 288).
    // attn_q is [n_embd, q_dim] in ggml column-major, so ne[1] = n_head * head_dim.
    const int q_dim   = (int)lw.attn_q->ne[1];           // total Q output size
    const int head_dim = q_dim / cfg.n_head;
    const int kv_dim   = (int)lw.attn_k->ne[1];          // total K/V output size
    const float scale  = 1.0f / sqrtf((float)head_dim);

    // ── 1. Attention layer norm ──────────────────────────────────────────────
    ggml_tensor* attn_norm_in = ggml_rms_norm(cctx, inpL, cfg.rms_norm_eps);
    // Note: GGUF converter already bakes +1 into Gemma norm weights, so just multiply.
    ggml_tensor* cur = ggml_mul(cctx, attn_norm_in, lw.attn_norm);

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

    // Gemma-2: attention logit soft-capping: KQ = cap * tanh(KQ / cap)
    if (cfg.attn_logit_cap > 0.0f) {
        KQ = ggml_scale(cctx, KQ, 1.0f / cfg.attn_logit_cap);
        KQ = ggml_tanh(cctx, KQ);
        KQ = ggml_scale(cctx, KQ, cfg.attn_logit_cap);
    }

    // Causal mask: prevent attending to future tokens
    KQ = ggml_diag_mask_inf(cctx, KQ, 0);
    KQ = ggml_soft_max(cctx, KQ);

    // Transpose V: [head_dim, n_tokens, n_kv_head] -> [n_tokens, head_dim, n_kv_head]
    ggml_tensor* V_t = ggml_cont(cctx, ggml_transpose(cctx, V_p));

    // KQV = V_t^T @ KQ -> [head_dim, n_tokens, n_head]
    ggml_tensor* KQV = ggml_mul_mat(cctx, V_t, KQ);
    KQV = ggml_permute(cctx, KQV, 0, 2, 1, 3);
    KQV = ggml_cont(cctx, KQV);
    KQV = ggml_reshape_2d(cctx, KQV, q_dim, n_tokens);

    // ── 5. Output projection + residual ──────────────────────────────────────
    ggml_tensor* attn_out = ggml_mul_mat(cctx, lw.attn_o, KQV);

    // Gemma-2: post-attention RMS norm
    if (cfg.use_post_attn_norm && lw.post_attn_norm) {
        attn_out = ggml_rms_norm(cctx, attn_out, cfg.rms_norm_eps);
        attn_out = ggml_mul(cctx, attn_out, lw.post_attn_norm);
    }

    ggml_tensor* inpFF = ggml_add(cctx, attn_out, inpL);

    // ── 6. FFN layer norm ─────────────────────────────────────────────────────
    ggml_tensor* ffn_norm_in = ggml_rms_norm(cctx, inpFF, cfg.rms_norm_eps);
    ggml_tensor* ff_cur = ggml_mul(cctx, ffn_norm_in, lw.ffn_norm);

    // ── 7. SwiGLU FFN ────────────────────────────────────────────────────────
    // Gemma-2 uses GELU instead of SiLU
    ggml_tensor* gate    = ggml_mul_mat(cctx, lw.ffn_gate, ff_cur);
    ggml_tensor* up      = ggml_mul_mat(cctx, lw.ffn_up,   ff_cur);
    gate                 = (cfg.attn_logit_cap > 0.0f) ? ggml_gelu(cctx, gate) : ggml_silu(cctx, gate);
    ggml_tensor* ffn_act = ggml_mul(cctx, gate, up);
    ggml_tensor* ffn_out = ggml_mul_mat(cctx, lw.ffn_down, ffn_act);

    // ── 8. Post-FFN norm (Gemma-2) ────────────────────────────────────────────
    if (cfg.use_post_ffn_norm && lw.post_ffn_norm) {
        ffn_out = ggml_rms_norm(cctx, ffn_out, cfg.rms_norm_eps);
        ffn_out = ggml_mul(cctx, ffn_out, lw.post_ffn_norm);
    }

    // ── 9. Residual ───────────────────────────────────────────────────────────
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
    const int n_layers = mc->block_end - mc->block_start + 1;

    // Working buffer — ping-pong between blocks
    std::vector<float> hidden((size_t)n_out);
    memcpy(hidden.data(), input_data, (size_t)n_out * sizeof(float));

    // Process each block in its own ggml context to avoid object-pool exhaustion.
    // A single context for all 26 Gemma-2 blocks creates ~1200 intermediate
    // tensors which overflows ggml's per-context memory pool.
    for (int i = 0; i < n_layers; i++) {
        // Scratch per single block — needs room for ~40 tensors + graph metadata.
        // ggml_new_graph alone allocates ~16 MB for GGML_DEFAULT_GRAPH_SIZE.
        size_t scratch_bytes = (size_t)n_embd * n_tokens * 200 + 64 * 1024 * 1024;
        std::vector<uint8_t> scratch(scratch_bytes);

        ggml_init_params cparams{ scratch.size(), scratch.data(), false };
        ggml_context* cctx = ggml_init(cparams);
        if (!cctx) throw std::runtime_error("block_runner_forward: ggml_init failed");

        // Input tensor [n_embd, n_tokens]
        ggml_tensor* inpL = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, n_embd, n_tokens);
        memcpy(inpL->data, hidden.data(), (size_t)n_out * sizeof(float));

        // Position tensor [n_tokens] i32
        ggml_tensor* pos = ggml_new_tensor_1d(cctx, GGML_TYPE_I32, n_tokens);
        int32_t* pos_data = (int32_t*)pos->data;
        for (int j = 0; j < n_tokens; j++) pos_data[j] = j;

        ggml_tensor* out = build_block(cctx, mc->layers[i], cfg, inpL, pos, n_tokens, mc->block_start + i);
        ggml_set_name(out, "block_out");

        ggml_cgraph* graph = ggml_new_graph(cctx);
        ggml_build_forward_expand(graph, out);

        ggml_status status = ggml_graph_compute_with_ctx(cctx, graph, g_n_threads);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_free(cctx);
            throw std::runtime_error("block_runner_forward: compute failed on block " +
                std::to_string(mc->block_start + i));
        }

        memcpy(hidden.data(), out->data, (size_t)n_out * sizeof(float));
        ggml_free(cctx);
    }

    return hidden;
}

// ── Embedding ─────────────────────────────────────────────────────────────────

std::vector<float> embed_tokens(
    ModelContext*  mc,
    const int32_t* token_ids,
    int            n_tokens
) {
    if (!mc->embd_weight)
        throw std::runtime_error("embed_tokens: no embedding weight loaded (block_start != 0?)");

    const int n_embd = mc->config.n_embd;
    const int n_out  = n_tokens * n_embd;

    // Small scratch: only need the get_rows result + token id tensor
    size_t scratch_bytes = (size_t)n_out * sizeof(float) + (size_t)n_tokens * sizeof(int32_t) + 4 * 1024 * 1024;
    std::vector<uint8_t> scratch(scratch_bytes);

    ggml_init_params cparams{ scratch.size(), scratch.data(), false };
    ggml_context* cctx = ggml_init(cparams);
    if (!cctx) throw std::runtime_error("embed_tokens: ggml_init failed");

    // Token IDs tensor [n_tokens] i32
    ggml_tensor* ids = ggml_new_tensor_1d(cctx, GGML_TYPE_I32, n_tokens);
    memcpy(ids->data, token_ids, (size_t)n_tokens * sizeof(int32_t));

    // Look up rows from embedding matrix: embd_weight is [n_embd, n_vocab]
    // ggml_get_rows extracts rows by index, returning [n_embd, n_tokens] f32
    ggml_tensor* embd = ggml_get_rows(cctx, mc->embd_weight, ids);

    // Gemma: scale embeddings by sqrt(n_embd)
    if (mc->config.embed_scale != 1.0f) {
        embd = ggml_scale(cctx, embd, mc->config.embed_scale);
    }

    ggml_set_name(embd, "embeddings");

    ggml_cgraph* graph = ggml_new_graph(cctx);
    ggml_build_forward_expand(graph, embd);

    ggml_status status = ggml_graph_compute_with_ctx(cctx, graph, g_n_threads);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(cctx);
        throw std::runtime_error("embed_tokens: ggml_graph_compute failed");
    }

    std::vector<float> result((size_t)n_out);
    memcpy(result.data(), embd->data, (size_t)n_out * sizeof(float));

    ggml_free(cctx);
    return result;
}

// ── Output projection ─────────────────────────────────────────────────────────

std::vector<float> project_to_logits(
    ModelContext* mc,
    const float*  hidden_states,
    int           n_tokens
) {
    if (!mc->output_norm)
        throw std::runtime_error("project_to_logits: no output_norm loaded (not hosting last block?)");
    if (!mc->output_wt)
        throw std::runtime_error("project_to_logits: no output_wt loaded (not hosting last block?)");

    const ModelConfig& cfg    = mc->config;
    const int          n_embd = cfg.n_embd;
    const int          n_vocab = cfg.n_vocab;
    const int          n_out  = n_tokens * n_vocab;

    // Scratch needs room for logits tensor [n_vocab, n_tokens], intermediate tensors,
    // graph metadata (~16 MB), and ggml_graph_compute_with_ctx's internal work buffer
    // (mul_mat with 256K vocab needs ~37 MB alone).
    size_t scratch_bytes = (size_t)n_out * sizeof(float) * 4 + (size_t)n_tokens * n_embd * sizeof(float) * 4 + 128 * 1024 * 1024;
    std::vector<uint8_t> scratch(scratch_bytes);

    ggml_init_params cparams{ scratch.size(), scratch.data(), false };
    ggml_context* cctx = ggml_init(cparams);
    if (!cctx) throw std::runtime_error("project_to_logits: ggml_init failed");

    // Input hidden states [n_embd, n_tokens]
    ggml_tensor* hidden = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(hidden, "hidden_in");
    memcpy(hidden->data, hidden_states, (size_t)n_tokens * n_embd * sizeof(float));

    // RMS norm (GGUF converter already bakes +1 into Gemma norm weights)
    ggml_tensor* normed = ggml_rms_norm(cctx, hidden, cfg.rms_norm_eps);
    normed = ggml_mul(cctx, normed, mc->output_norm);

    // Project to vocab: output_wt is [n_embd, n_vocab] → result is [n_vocab, n_tokens]
    ggml_tensor* logits = ggml_mul_mat(cctx, mc->output_wt, normed);

    // Gemma-2: final logit soft-capping: logits = cap * tanh(logits / cap)
    if (cfg.final_logit_cap > 0.0f) {
        logits = ggml_scale(cctx, logits, 1.0f / cfg.final_logit_cap);
        logits = ggml_tanh(cctx, logits);
        logits = ggml_scale(cctx, logits, cfg.final_logit_cap);
    }

    ggml_set_name(logits, "logits");

    ggml_cgraph* graph = ggml_new_graph(cctx);
    ggml_build_forward_expand(graph, logits);

    ggml_status status = ggml_graph_compute_with_ctx(cctx, graph, g_n_threads);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(cctx);
        throw std::runtime_error("project_to_logits: ggml_graph_compute failed");
    }

    std::vector<float> result((size_t)n_out);
    memcpy(result.data(), logits->data, (size_t)n_out * sizeof(float));

    ggml_free(cctx);
    return result;
}

// ── KV-cached inference sessions ──────────────────────────────────────────────

int create_session(ModelContext* mc, int max_length) {
    if (max_length <= 0)
        throw std::runtime_error("create_session: max_length must be > 0");

    const int n_layers = mc->block_end - mc->block_start + 1;
    const auto& lw0 = mc->layers[0];
    const int head_dim  = (int)lw0.attn_k->ne[0];   // ne[0] of K weight = head_dim
    const int n_kv_head = mc->config.n_kv_head;

    auto* sess = new SessionState();
    sess->max_length    = max_length;
    sess->prefix_length = 0;
    sess->head_dim      = head_dim;
    sess->n_kv_head     = n_kv_head;
    sess->n_layers      = n_layers;

    size_t per_layer = (size_t)head_dim * n_kv_head * max_length;
    sess->k_cache.resize(n_layers);
    sess->v_cache.resize(n_layers);
    for (int i = 0; i < n_layers; i++) {
        sess->k_cache[i].resize(per_layer, 0.0f);
        sess->v_cache[i].resize(per_layer, 0.0f);
    }

    int id = mc->next_session_id++;
    mc->sessions[id] = sess;
    return id;
}

void free_session(ModelContext* mc, int session_id) {
    auto it = mc->sessions.find(session_id);
    if (it != mc->sessions.end()) {
        delete it->second;
        mc->sessions.erase(it);
    }
}

std::vector<float> session_forward(
    ModelContext* mc,
    int           session_id,
    const float*  input_data,
    int           n_new_tokens
) {
    auto it = mc->sessions.find(session_id);
    if (it == mc->sessions.end())
        throw std::runtime_error("session_forward: invalid session_id");
    SessionState* sess = it->second;

    const ModelConfig& cfg    = mc->config;
    const int          n_embd = cfg.n_embd;
    const int          n_out  = n_new_tokens * n_embd;
    const int          n_layers = mc->block_end - mc->block_start + 1;
    const int          total_seq = sess->prefix_length + n_new_tokens;

    if (total_seq > sess->max_length)
        throw std::runtime_error("session_forward: sequence exceeds max_length");

    std::vector<float> hidden((size_t)n_out);
    memcpy(hidden.data(), input_data, (size_t)n_out * sizeof(float));

    for (int li = 0; li < n_layers; li++) {
        const auto& lw = mc->layers[li];

        const int q_dim    = (int)lw.attn_q->ne[1];
        const int head_dim = q_dim / cfg.n_head;
        const int n_kv_head = cfg.n_kv_head;
        const float scale  = 1.0f / sqrtf((float)head_dim);

        // Scratch: room for new tokens, cached K/V copies, concat, attention, FFN, graph
        size_t cache_bytes = (size_t)sess->prefix_length * head_dim * n_kv_head * sizeof(float) * 4;
        size_t scratch_bytes = cache_bytes
            + (size_t)n_embd * n_new_tokens * 200
            + 64 * 1024 * 1024;
        std::vector<uint8_t> scratch(scratch_bytes);

        ggml_init_params cparams{ scratch.size(), scratch.data(), false };
        ggml_context* cctx = ggml_init(cparams);
        if (!cctx) throw std::runtime_error("session_forward: ggml_init failed");

        // ── Inputs ───────────────────────────────────────────────────────────
        ggml_tensor* inpL = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, n_embd, n_new_tokens);
        memcpy(inpL->data, hidden.data(), (size_t)n_out * sizeof(float));

        ggml_tensor* pos = ggml_new_tensor_1d(cctx, GGML_TYPE_I32, n_new_tokens);
        {
            int32_t* pd = (int32_t*)pos->data;
            for (int j = 0; j < n_new_tokens; j++) pd[j] = sess->prefix_length + j;
        }

        // ── 1. Attention pre-norm ────────────────────────────────────────────
        ggml_tensor* cur = ggml_mul(cctx,
            ggml_rms_norm(cctx, inpL, cfg.rms_norm_eps), lw.attn_norm);

        // ── 2. QKV projections + reshape + RoPE ──────────────────────────────
        ggml_tensor* Qcur = ggml_reshape_3d(cctx,
            ggml_mul_mat(cctx, lw.attn_q, cur), head_dim, cfg.n_head, n_new_tokens);
        ggml_tensor* Kcur = ggml_reshape_3d(cctx,
            ggml_mul_mat(cctx, lw.attn_k, cur), head_dim, n_kv_head, n_new_tokens);
        ggml_tensor* Vcur = ggml_reshape_3d(cctx,
            ggml_mul_mat(cctx, lw.attn_v, cur), head_dim, n_kv_head, n_new_tokens);

        Qcur = ggml_rope_ext(cctx, Qcur, pos, nullptr,
            head_dim, 0, 0, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        Kcur = ggml_rope_ext(cctx, Kcur, pos, nullptr,
            head_dim, 0, 0, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Remember K/V data pointers — we'll copy to cache after eval
        float* k_new_ptr = (float*)Kcur->data;
        // For V: reshape is a view sharing data with the matmul output.
        // We need the raw V projection pointer (the matmul result), which
        // Vcur->data already points to (reshape preserves the pointer).
        float* v_new_ptr = (float*)Vcur->data;

        // ── 3. Combine with cached K/V ───────────────────────────────────────
        ggml_tensor* K_full;
        ggml_tensor* V_full;

        if (sess->prefix_length > 0) {
            ggml_tensor* K_cached = ggml_new_tensor_3d(cctx, GGML_TYPE_F32,
                head_dim, n_kv_head, sess->prefix_length);
            memcpy(K_cached->data, sess->k_cache[li].data(),
                (size_t)head_dim * n_kv_head * sess->prefix_length * sizeof(float));

            ggml_tensor* V_cached = ggml_new_tensor_3d(cctx, GGML_TYPE_F32,
                head_dim, n_kv_head, sess->prefix_length);
            memcpy(V_cached->data, sess->v_cache[li].data(),
                (size_t)head_dim * n_kv_head * sess->prefix_length * sizeof(float));

            K_full = ggml_concat(cctx, K_cached, Kcur, 2);
            V_full = ggml_concat(cctx, V_cached, Vcur, 2);
        } else {
            K_full = Kcur;
            V_full = Vcur;
        }

        // ── 4. Scaled dot-product attention ──────────────────────────────────
        ggml_tensor* Q_p = ggml_permute(cctx, Qcur,  0, 2, 1, 3);
        ggml_tensor* K_p = ggml_permute(cctx, K_full, 0, 2, 1, 3);
        ggml_tensor* V_p = ggml_permute(cctx, V_full, 0, 2, 1, 3);

        ggml_tensor* KQ = ggml_mul_mat(cctx, K_p, Q_p);
        KQ = ggml_scale(cctx, KQ, scale);

        if (cfg.attn_logit_cap > 0.0f) {
            KQ = ggml_scale(cctx, KQ, 1.0f / cfg.attn_logit_cap);
            KQ = ggml_tanh(cctx, KQ);
            KQ = ggml_scale(cctx, KQ, cfg.attn_logit_cap);
        }

        KQ = ggml_diag_mask_inf(cctx, KQ, sess->prefix_length);
        KQ = ggml_soft_max(cctx, KQ);

        ggml_tensor* V_t  = ggml_cont(cctx, ggml_transpose(cctx, V_p));
        ggml_tensor* KQV  = ggml_mul_mat(cctx, V_t, KQ);
        KQV = ggml_permute(cctx, KQV, 0, 2, 1, 3);
        KQV = ggml_cont(cctx, KQV);
        KQV = ggml_reshape_2d(cctx, KQV, q_dim, n_new_tokens);

        // ── 5. Output projection + post-attn norm + residual ─────────────────
        ggml_tensor* attn_out = ggml_mul_mat(cctx, lw.attn_o, KQV);
        if (cfg.use_post_attn_norm && lw.post_attn_norm) {
            attn_out = ggml_rms_norm(cctx, attn_out, cfg.rms_norm_eps);
            attn_out = ggml_mul(cctx, attn_out, lw.post_attn_norm);
        }
        ggml_tensor* inpFF = ggml_add(cctx, attn_out, inpL);

        // ── 6. FFN norm ──────────────────────────────────────────────────────
        ggml_tensor* ff_cur = ggml_mul(cctx,
            ggml_rms_norm(cctx, inpFF, cfg.rms_norm_eps), lw.ffn_norm);

        // ── 7. Gated FFN ────────────────────────────────────────────────────
        ggml_tensor* gate = ggml_mul_mat(cctx, lw.ffn_gate, ff_cur);
        ggml_tensor* up   = ggml_mul_mat(cctx, lw.ffn_up,   ff_cur);
        gate = (cfg.attn_logit_cap > 0.0f) ? ggml_gelu(cctx, gate)
                                           : ggml_silu(cctx, gate);
        ggml_tensor* ffn_out = ggml_mul_mat(cctx, lw.ffn_down,
            ggml_mul(cctx, gate, up));

        // ── 8. Post-FFN norm + residual ──────────────────────────────────────
        if (cfg.use_post_ffn_norm && lw.post_ffn_norm) {
            ffn_out = ggml_rms_norm(cctx, ffn_out, cfg.rms_norm_eps);
            ffn_out = ggml_mul(cctx, ffn_out, lw.post_ffn_norm);
        }
        ggml_tensor* out = ggml_add(cctx, ffn_out, inpFF);

        // ── Build & evaluate ─────────────────────────────────────────────────
        ggml_cgraph* graph = ggml_new_graph(cctx);
        ggml_build_forward_expand(graph, out);

        ggml_status status = ggml_graph_compute_with_ctx(cctx, graph, g_n_threads);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_free(cctx);
            throw std::runtime_error("session_forward: compute failed at block " +
                std::to_string(mc->block_start + li));
        }

        // ── Update KV cache ─────────────────────────────────────────────────
        size_t kv_new_floats = (size_t)head_dim * n_kv_head * n_new_tokens;
        size_t kv_offset     = (size_t)head_dim * n_kv_head * sess->prefix_length;
        memcpy(sess->k_cache[li].data() + kv_offset, k_new_ptr,
            kv_new_floats * sizeof(float));
        memcpy(sess->v_cache[li].data() + kv_offset, v_new_ptr,
            kv_new_floats * sizeof(float));

        // ── Copy output hidden ───────────────────────────────────────────────
        memcpy(hidden.data(), out->data, (size_t)n_out * sizeof(float));
        ggml_free(cctx);
    }

    sess->prefix_length += n_new_tokens;
    return hidden;
}
