#include "model_context.h"
#include "ggml.h"
#include "gguf.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

// ── Metadata helpers ──────────────────────────────────────────────────────────

static int32_t meta_u32(gguf_context* g, const char* key, int32_t def = 0) {
    int64_t i = gguf_find_key(g, key);
    return (i >= 0) ? (int32_t)gguf_get_val_u32(g, i) : def;
}

static float meta_f32(gguf_context* g, const char* key, float def = 0.0f) {
    int64_t i = gguf_find_key(g, key);
    return (i >= 0) ? gguf_get_val_f32(g, i) : def;
}

static std::string meta_str(gguf_context* g, const char* key, const char* def = "") {
    int64_t i = gguf_find_key(g, key);
    return (i >= 0) ? gguf_get_val_str(g, i) : def;
}

/** Read a u32 metadata value, trying the architecture-specific key first,
 *  then falling back to llama.* for compatibility. */
static int32_t arch_u32(gguf_context* g, const std::string& arch, const char* suffix, int32_t def = 0) {
    int32_t v = meta_u32(g, (arch + "." + suffix).c_str(), -1);
    if (v != -1) return v;
    return meta_u32(g, (std::string("llama.") + suffix).c_str(), def);
}

static float arch_f32(gguf_context* g, const std::string& arch, const char* suffix, float def = 0.0f) {
    float v = meta_f32(g, (arch + "." + suffix).c_str(), -1.0f);
    if (v != -1.0f) return v;
    return meta_f32(g, (std::string("llama.") + suffix).c_str(), def);
}

// ── Tensor lookup + data pointer fixup ────────────────────────────────────────
// Finds tensor by name in ggml_context (created by gguf_init), then sets its
// .data to point into file_buf at the correct offset (zero-copy).
static ggml_tensor* find_and_bind(
    ggml_context* wctx,
    gguf_context* gctx,
    const char*   name,
    uint8_t*      file_buf,
    size_t        data_region_start
) {
    ggml_tensor* t = ggml_get_tensor(wctx, name);
    if (!t) return nullptr;

    int64_t tidx = gguf_find_tensor(gctx, name);
    if (tidx < 0) return nullptr;

    size_t off = gguf_get_tensor_offset(gctx, tidx);
    t->data = file_buf + data_region_start + off;
    return t;
}

// ── Public API ────────────────────────────────────────────────────────────────

ModelContext* model_context_load(
    const char* model_path,
    int         block_start,
    int         block_end,
    int         total_blocks
) {
    // Validate parameters
    if (block_start < 0 || block_end < block_start || total_blocks <= 0 || block_end >= total_blocks) {
        throw std::runtime_error("Invalid block range: block_start=" + std::to_string(block_start) +
            " block_end=" + std::to_string(block_end) + " total_blocks=" + std::to_string(total_blocks));
    }

    // 1. Read entire GGUF file into memory
    FILE* f = fopen(model_path, "rb");
    if (!f) throw std::runtime_error(std::string("Cannot open: ") + model_path);

    fseek(f, 0, SEEK_END);
    size_t fsize = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);

    std::vector<uint8_t> buf(fsize);
    if (fread(buf.data(), 1, fsize, f) != fsize) {
        fclose(f);
        throw std::runtime_error("Short read on model file");
    }
    fclose(f);

    // 2. Parse GGUF header and create ggml tensors (no_alloc: data not allocated)
    ggml_context* wctx = nullptr;
    gguf_init_params gparams;
    gparams.no_alloc = true;
    gparams.ctx      = &wctx;

    gguf_context* gctx = gguf_init_from_file(model_path, gparams);
    if (!gctx) {
        std::string msg = "Failed to parse GGUF header for: ";
        msg += model_path;
        msg += " (file size: " + std::to_string(fsize) + " bytes)";
        throw std::runtime_error(msg);
    }
    struct GufGuard { gguf_context* g; ~GufGuard() { if (g) gguf_free(g); } } gctx_guard{gctx};
    if (!wctx) { throw std::runtime_error("gguf_init did not create ggml context"); }
    struct GmlGuard { ggml_context* g; ~GmlGuard() { if (g) ggml_free(g); } } wctx_guard{wctx};

    size_t data_start = gguf_get_data_offset(gctx);

    auto* mc = new ModelContext();
    struct McGuard { ModelContext* p; ~McGuard() { if (p) { if (p->weight_ctx) { ggml_free(p->weight_ctx); p->weight_ctx = nullptr; } delete p; } } } mc_guard{mc};
    mc->file_buf    = std::move(buf);
    mc->weight_ctx  = wctx;
    mc->block_start = block_start;
    mc->block_end   = block_end;

    uint8_t* fb = mc->file_buf.data();

    // 3. Read model config from GGUF metadata
    std::string arch = meta_str(gctx, "general.architecture", "llama");
    mc->config.n_embd         = arch_u32(gctx, arch, "embedding_length");
    mc->config.n_head         = arch_u32(gctx, arch, "attention.head_count");
    mc->config.n_kv_head      = arch_u32(gctx, arch, "attention.head_count_kv");
    mc->config.n_ff           = arch_u32(gctx, arch, "feed_forward_length");
    mc->config.rms_norm_eps   = arch_f32(gctx, arch, "attention.layer_norm_rms_epsilon", 1e-5f);
    mc->config.rope_freq_base = arch_f32(gctx, arch, "rope.freq_base", 10000.0f);

    // Architecture-specific flags
    bool is_gemma = (arch == "gemma" || arch == "gemma2");
    if (is_gemma) {
        mc->config.embed_scale       = sqrtf((float)mc->config.n_embd);
        mc->config.norm_weight_plus1 = true;
    }
    if (arch == "gemma2") {
        // Gemma-2 2B uses logit soft-capping
        mc->config.attn_logit_cap    = arch_f32(gctx, arch, "attn_logit_softcapping", 50.0f);
        mc->config.final_logit_cap   = arch_f32(gctx, arch, "final_logit_softcapping", 30.0f);
        mc->config.use_post_attn_norm = true;
        mc->config.use_post_ffn_norm  = true;
    }

    // 4. Load embedding tensor if we host block 0
    if (block_start == 0) {
        mc->embd_weight = find_and_bind(wctx, gctx, "token_embd.weight", fb, data_start);
        if (mc->embd_weight) mc->config.n_vocab = (int32_t)mc->embd_weight->ne[1];
    }

    // 5. Load transformer block tensors
    int n_layers = block_end - block_start + 1;
    mc->layers.resize(n_layers);
    for (int bi = block_start; bi <= block_end; bi++) {
        auto& lw  = mc->layers[bi - block_start];
        auto  pfx = "blk." + std::to_string(bi) + ".";
        auto  B   = [&](const char* suf) {
            return find_and_bind(wctx, gctx, (pfx + suf).c_str(), fb, data_start);
        };
        lw.attn_norm = B("attn_norm.weight");
        lw.attn_q    = B("attn_q.weight");
        lw.attn_k    = B("attn_k.weight");
        lw.attn_v    = B("attn_v.weight");
        lw.attn_o    = B("attn_output.weight");
        lw.ffn_norm  = B("ffn_norm.weight");
        lw.ffn_gate  = B("ffn_gate.weight");
        lw.ffn_up    = B("ffn_up.weight");
        lw.ffn_down  = B("ffn_down.weight");
        // Post-norms (Gemma-2)
        if (mc->config.use_post_attn_norm) lw.post_attn_norm = B("post_attention_norm.weight");
        if (mc->config.use_post_ffn_norm)  lw.post_ffn_norm  = B("post_ffw_norm.weight");
    }

    // 6. Load output tensors if we host the last block
    if (block_end == total_blocks - 1) {
        mc->output_norm = find_and_bind(wctx, gctx, "output_norm.weight", fb, data_start);
        mc->output_wt   = find_and_bind(wctx, gctx, "output.weight",      fb, data_start);

        // Weight tying: many models (Gemma, etc.) share token_embd.weight as
        // the output projection — no separate output.weight tensor exists.
        if (!mc->output_wt && mc->embd_weight) {
            mc->output_wt = mc->embd_weight;
        }
        // If we don't host block 0, embd_weight wasn't loaded yet — load it now
        // solely for the output projection.
        if (!mc->output_wt) {
            mc->output_wt = find_and_bind(wctx, gctx, "token_embd.weight", fb, data_start);
        }

        if (mc->output_wt && mc->config.n_vocab == 0)
            mc->config.n_vocab = (int32_t)mc->output_wt->ne[1];
    }

    // gctx_guard fires here and frees gctx (mc does not own it)
    wctx_guard.g = nullptr;  // disarm — mc now owns wctx cleanup via weight_ctx
    mc_guard.p   = nullptr;  // disarm — caller takes ownership of mc
    return mc;
}

void model_context_free(ModelContext* mc) {
    if (!mc) return;
    for (auto& pair : mc->sessions) delete pair.second;
    mc->sessions.clear();
    if (mc->weight_ctx) ggml_free(mc->weight_ctx);
    delete mc;
}
