#include "model_context.h"
#include "ggml.h"
#include "gguf.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

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
    std::string key = arch + "." + suffix;
    int64_t i = gguf_find_key(g, key.c_str());
    if (i >= 0) return (int32_t)gguf_get_val_u32(g, i);
    std::string fallback = std::string("llama.") + suffix;
    i = gguf_find_key(g, fallback.c_str());
    if (i >= 0) return (int32_t)gguf_get_val_u32(g, i);
    return def;
}

static float arch_f32(gguf_context* g, const std::string& arch, const char* suffix, float def = 0.0f) {
    std::string key = arch + "." + suffix;
    int64_t i = gguf_find_key(g, key.c_str());
    if (i >= 0) return gguf_get_val_f32(g, i);
    std::string fallback = std::string("llama.") + suffix;
    i = gguf_find_key(g, fallback.c_str());
    if (i >= 0) return gguf_get_val_f32(g, i);
    return def;
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

// Multi-shard version: searches across all shard (wctx, gctx, buf) triples.
static ggml_tensor* find_and_bind_shards(
    const std::vector<ggml_context*>&          wctxs,
    const std::vector<gguf_context*>&          gctxs,
    const std::vector<std::vector<uint8_t>>&   bufs,
    const std::vector<size_t>&                 data_starts,
    const char* name
) {
    for (size_t i = 0; i < wctxs.size(); i++) {
        ggml_tensor* t = ggml_get_tensor(wctxs[i], name);
        if (!t) continue;
        int64_t tidx = gguf_find_tensor(gctxs[i], name);
        if (tidx < 0) continue;
        size_t off = gguf_get_tensor_offset(gctxs[i], tidx);
        t->data = const_cast<uint8_t*>(bufs[i].data()) + data_starts[i] + off;
        return t;
    }
    return nullptr;
}

// ── Public API ────────────────────────────────────────────────────────────────

ModelContext* model_context_load(
    const char* model_path,
    int         block_start,
    int         block_end,
    int         total_blocks
) {
    // Validate parameters — block_end == -1 is a "shim mode" sentinel (no blocks, embed+output only)
    bool shim_mode = (block_end == -1);
    if (!shim_mode && (block_start < 0 || block_end < block_start || total_blocks <= 0 || block_end >= total_blocks)) {
        throw std::runtime_error("Invalid block range: block_start=" + std::to_string(block_start) +
            " block_end=" + std::to_string(block_end) + " total_blocks=" + std::to_string(total_blocks));
    }
    if (shim_mode && (total_blocks <= 0 || block_start != 0)) {
        throw std::runtime_error("Shim mode requires block_start=0 and total_blocks>0, got block_start=" +
            std::to_string(block_start) + " total_blocks=" + std::to_string(total_blocks));
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
    struct McGuard { ModelContext* p; ~McGuard() { if (p) { for (auto w : p->shard_wctxs) if (w) ggml_free(w); delete p; } } } mc_guard{mc};
    mc->shard_bufs.push_back(std::move(buf));
    mc->shard_wctxs.push_back(nullptr);  // may throw — wctx_guard still armed
    mc->shard_wctxs.back() = wctx;       // infallible pointer assignment
    wctx_guard.g = nullptr;              // disarm — McGuard owns wctx through shard_wctxs
    mc->block_start = block_start;
    mc->block_end   = block_end;

    uint8_t* fb = mc->shard_bufs[0].data();

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

    // 5. Load transformer block tensors (skipped in shim mode)
    int n_layers = shim_mode ? 0 : (block_end - block_start + 1);
    mc->layers.resize(n_layers);
    for (int bi = block_start; !shim_mode && bi <= block_end; bi++) {
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
    if (shim_mode || block_end == total_blocks - 1) {
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
    mc_guard.p = nullptr;  // disarm — caller takes ownership of mc
    return mc;
}

void model_context_free(ModelContext* mc) {
    if (!mc) return;
    for (auto& pair : mc->sessions) delete pair.second;
    mc->sessions.clear();
    for (auto wctx : mc->shard_wctxs) {
        if (wctx) ggml_free(wctx);
    }
    delete mc;
}

ModelContext* model_context_load_shards(
    const char** shard_paths,
    int          n_shards,
    int          block_start,
    int          block_end,
    int          total_blocks
) {
    if (n_shards < 1) throw std::runtime_error("model_context_load_shards: n_shards < 1");
    if (block_start < 0 || block_end < block_start || total_blocks <= 0 || block_end >= total_blocks)
        throw std::runtime_error("Invalid block range");

    auto* mc = new ModelContext();
    struct McGuard {
        ModelContext* p;
        ~McGuard() { if (p) { for (auto w : p->shard_wctxs) if (w) ggml_free(w); delete p; } }
    } mc_guard{mc};

    mc->block_start = block_start;
    mc->block_end   = block_end;

    std::vector<gguf_context*> gctxs;
    std::vector<size_t>        data_starts;

    // Declared before the loop so its destructor covers all elements pushed so far
    // if the loop throws mid-way.
    struct GctxGuard {
        std::vector<gguf_context*>& gs;
        ~GctxGuard() { for (auto g : gs) if (g) gguf_free(g); }
    } gctx_guard{gctxs};

    for (int si = 0; si < n_shards; si++) {
        FILE* f = fopen(shard_paths[si], "rb");
        if (!f) throw std::runtime_error(std::string("Cannot open shard: ") + shard_paths[si]);
        fseek(f, 0, SEEK_END);
        size_t fsize = (size_t)ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<uint8_t> buf(fsize);
        if (fread(buf.data(), 1, fsize, f) != fsize) {
            fclose(f);
            throw std::runtime_error("Short read on shard");
        }
        fclose(f);

        ggml_context* wctx = nullptr;
        gguf_init_params gp;
        gp.no_alloc = true;
        gp.ctx      = &wctx;
        gguf_context* gctx = gguf_init_from_file(shard_paths[si], gp);
        if (!gctx) throw std::runtime_error(std::string("Failed to parse GGUF shard: ") + shard_paths[si]);
        if (!wctx) { gguf_free(gctx); throw std::runtime_error("gguf_init did not create ggml context"); }

        // Push gctx first so GctxGuard covers it on any subsequent throw.
        gctxs.push_back(gctx);

        // Local guard for wctx until McGuard (via shard_wctxs) takes ownership.
        // push_back(nullptr) may throw; if so WctxGuard frees wctx. The
        // subsequent .back() = wctx is infallible (pointer assignment), so
        // we can safely disarm the guard immediately after.
        struct WctxGuard {
            ggml_context*& w;
            ~WctxGuard() { if (w) { ggml_free(w); w = nullptr; } }
        } wctx_guard{wctx};

        mc->shard_wctxs.push_back(nullptr);  // reserve slot first — may throw; WctxGuard covers wctx
        mc->shard_bufs.push_back(std::move(buf));  // push buf after slot reserved
        mc->shard_wctxs.back() = wctx;       // infallible pointer assignment
        wctx_guard.w = nullptr;              // disarm — McGuard owns wctx through shard_wctxs

        data_starts.push_back(gguf_get_data_offset(gctx));
    }

    // Read config from shard 0 (has all metadata KVs)
    gguf_context* g0 = gctxs[0];
    std::string arch = meta_str(g0, "general.architecture", "llama");
    mc->config.n_embd         = arch_u32(g0, arch, "embedding_length");
    mc->config.n_head         = arch_u32(g0, arch, "attention.head_count");
    mc->config.n_kv_head      = arch_u32(g0, arch, "attention.head_count_kv");
    mc->config.n_ff           = arch_u32(g0, arch, "feed_forward_length");
    mc->config.rms_norm_eps   = arch_f32(g0, arch, "attention.layer_norm_rms_epsilon", 1e-5f);
    mc->config.rope_freq_base = arch_f32(g0, arch, "rope.freq_base", 10000.0f);

    bool is_gemma = (arch == "gemma" || arch == "gemma2");
    if (is_gemma) {
        mc->config.embed_scale       = sqrtf((float)mc->config.n_embd);
        mc->config.norm_weight_plus1 = true;
    }
    if (arch == "gemma2") {
        mc->config.attn_logit_cap    = arch_f32(g0, arch, "attn_logit_softcapping", 50.0f);
        mc->config.final_logit_cap   = arch_f32(g0, arch, "final_logit_softcapping", 30.0f);
        mc->config.use_post_attn_norm = true;
        mc->config.use_post_ffn_norm  = true;
    }

    // Build tensor name → (shard_index, tensor*) map for O(1) per-tensor lookup.
    // Without this, find_and_bind_shards would be O(shards × tensors) per call.
    std::unordered_map<std::string, std::pair<size_t, ggml_tensor*>> tensor_map;
    for (size_t i = 0; i < mc->shard_wctxs.size(); i++) {
        int64_t n = gguf_get_n_tensors(gctxs[i]);
        for (int64_t j = 0; j < n; j++) {
            const char* tname = gguf_get_tensor_name(gctxs[i], j);
            ggml_tensor* t = ggml_get_tensor(mc->shard_wctxs[i], tname);
            if (t) tensor_map.emplace(tname, std::make_pair(i, t));
        }
    }

    auto B = [&](const char* name) -> ggml_tensor* {
        auto it = tensor_map.find(name);
        if (it == tensor_map.end()) return nullptr;
        size_t i = it->second.first;
        ggml_tensor* t = it->second.second;
        int64_t tidx = gguf_find_tensor(gctxs[i], name);
        if (tidx < 0) return nullptr;
        size_t off = gguf_get_tensor_offset(gctxs[i], tidx);
        t->data = const_cast<uint8_t*>(mc->shard_bufs[i].data()) + data_starts[i] + off;
        return t;
    };

    if (block_start == 0) {
        mc->embd_weight = B("token_embd.weight");
        if (mc->embd_weight) mc->config.n_vocab = (int32_t)mc->embd_weight->ne[1];
    }

    int n_layers = block_end - block_start + 1;
    mc->layers.resize(n_layers);
    for (int bi = block_start; bi <= block_end; bi++) {
        auto& lw  = mc->layers[bi - block_start];
        auto  pfx = std::string("blk.") + std::to_string(bi) + ".";
        auto  BL  = [&](const char* suf) -> ggml_tensor* {
            return B((pfx + suf).c_str());
        };
        lw.attn_norm = BL("attn_norm.weight");
        lw.attn_q    = BL("attn_q.weight");
        lw.attn_k    = BL("attn_k.weight");
        lw.attn_v    = BL("attn_v.weight");
        lw.attn_o    = BL("attn_output.weight");
        lw.ffn_norm  = BL("ffn_norm.weight");
        lw.ffn_gate  = BL("ffn_gate.weight");
        lw.ffn_up    = BL("ffn_up.weight");
        lw.ffn_down  = BL("ffn_down.weight");
        if (mc->config.use_post_attn_norm) lw.post_attn_norm = BL("post_attention_norm.weight");
        if (mc->config.use_post_ffn_norm)  lw.post_ffn_norm  = BL("post_ffw_norm.weight");
    }

    if (block_end == total_blocks - 1) {
        mc->output_norm = B("output_norm.weight");
        mc->output_wt   = B("output.weight");
        if (!mc->output_wt && mc->embd_weight) mc->output_wt = mc->embd_weight;
        if (!mc->output_wt) mc->output_wt = B("token_embd.weight");
        if (mc->output_wt && mc->config.n_vocab == 0)
            mc->config.n_vocab = (int32_t)mc->output_wt->ne[1];
    }

    mc_guard.p = nullptr;
    return mc;
}
