#include "llama_block_context.h"
#include <stdexcept>
#include <cstring>
#include <string>
#include <thread>
#include <algorithm>

static const int g_n_threads = [] {
    int hw = (int)std::thread::hardware_concurrency();
    return std::max(1, hw > 2 ? hw - 2 : 1);
}();

static constexpr int LBC_DEFAULT_CTX = 8192;

static LlamaBlockContext* lbc_load_internal(
        const char** paths, int n_shards,
        int32_t block_start, int32_t block_end, int32_t total_blocks) {
    llama_model_params mp  = llama_model_default_params();
    mp.block_range_start   = block_start;
    mp.block_range_end     = block_end;
    mp.n_gpu_layers        = 0;
    mp.use_mmap            = true;

    llama_model* model = (n_shards == 1)
        ? llama_model_load_from_file(paths[0], mp)
        : llama_model_load_from_splits(paths, (size_t)n_shards, mp);
    if (!model) throw std::runtime_error("llama_model_load failed — check GGUF path and architecture support");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = LBC_DEFAULT_CTX;
    cp.n_batch         = LBC_DEFAULT_CTX;
    cp.n_seq_max       = 64; // seq 0 reserved for stateless; sessions use 1..63
    cp.n_threads       = g_n_threads;
    cp.n_threads_batch = g_n_threads;
    cp.offload_kqv     = false;
    // embeddings=true makes llama_context store per-token embeddings after each decode,
    // which is required for get_embeddings_ith() in the patch 2 I/O functions.
    // Trade-off: slight overhead per batch vs. standard decode mode; unavoidable here.
    cp.embeddings      = true;

    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        throw std::runtime_error("llama_init_from_model failed");
    }

    auto* lbc         = new LlamaBlockContext();
    lbc->model        = model;
    lbc->ctx          = ctx;
    lbc->block_start  = block_start;
    lbc->block_end    = block_end;
    lbc->total_blocks = total_blocks;
    return lbc;
}

LlamaBlockContext* lbc_load(
        const char* path,
        int32_t block_start, int32_t block_end, int32_t total_blocks) {
    const char* paths[1] = { path };
    return lbc_load_internal(paths, 1, block_start, block_end, total_blocks);
}

LlamaBlockContext* lbc_load_sharded(
        const char** paths, int n_shards,
        int32_t block_start, int32_t block_end, int32_t total_blocks) {
    return lbc_load_internal(paths, n_shards, block_start, block_end, total_blocks);
}

void lbc_free(LlamaBlockContext* lbc) {
    if (!lbc) return;
    if (!lbc->sessions.empty())
        fprintf(stderr, "lbc_free: warning: %zu session(s) still open at free time\n", lbc->sessions.size());
    if (lbc->ctx)   llama_free(lbc->ctx);
    if (lbc->model) llama_model_free(lbc->model);
    delete lbc;
}

std::vector<float> lbc_embed_tokens(
        LlamaBlockContext* lbc, const int32_t* ids, int n_tokens) {
    if (n_tokens <= 0)
        throw std::runtime_error("lbc_embed_tokens: n_tokens must be > 0");
    if (lbc->block_start != 0)
        throw std::runtime_error("lbc_embed_tokens: only valid on first-node contexts (block_start == 0)");
    const int32_t n_embd = llama_model_n_embd(lbc->model);
    std::vector<float> out((size_t)n_embd * n_tokens);

    llama_memory_seq_rm(llama_get_memory(lbc->ctx), 0, -1, -1);

    // Use llama_batch_init (not llama_batch_get_one) so ownership is consistent with
    // lbc_forward and lbc_project_to_logits — all three paths own their batch and free it.
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]     = ids[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        // Shim mode (block_end==-1): the full model runs here; request logits for
        // the last token so projectToLogits can return them without re-decoding.
        batch.logits[i]    = (lbc->block_end == -1 && i == n_tokens - 1) ? 1 : 0;
    }
    batch.n_tokens = n_tokens;

    int rc = llama_get_hidden_from_tokens(lbc->ctx, batch, out.data());
    llama_batch_free(batch);
    if (rc != 0)
        throw std::runtime_error("llama_get_hidden_from_tokens failed: rc=" + std::to_string(rc));

    // Shim mode: cp.embeddings=true triggers output_all, so all tokens get logits and the
    // buffer is n_tokens*n_vocab floats.  Capture the LAST token's slice now while we know
    // n_tokens — lbc_project_to_logits cannot recompute this offset later.
    if (lbc->block_end == -1) {
        const int32_t n_vocab_loc = llama_vocab_n_tokens(llama_model_get_vocab(lbc->model));
        const float* raw = llama_get_logits(lbc->ctx);
        if (!raw)
            throw std::runtime_error("lbc_embed_tokens (shim): llama_get_logits returned null — logit flag not set?");
        lbc->shim_logits.assign(
            raw + (size_t)(n_tokens - 1) * n_vocab_loc,
            raw + (size_t)n_tokens       * n_vocab_loc
        );
    }

    return out;
}

// Always returns n_vocab logits for the last token only, regardless of n_tokens.
// Only valid when this context hosts the final block (block_end == total_blocks - 1).
std::vector<float> lbc_project_to_logits(
        LlamaBlockContext* lbc, const float* hidden, int n_tokens) {
    if (n_tokens <= 0)
        throw std::runtime_error("lbc_project_to_logits: n_tokens must be > 0");
    // block_end == -1 is the shim-runner sentinel (embed+project only, no forward blocks).
    // Both -1 and total_blocks-1 mean the full output tensors are present.
    if (lbc->block_end != lbc->total_blocks - 1 && lbc->block_end != -1)
        throw std::runtime_error("lbc_project_to_logits: only valid on last-node or shim contexts (block_end == total_blocks - 1 or -1)");
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(lbc->model));

    // Shim mode: lbc_embed_tokens already ran the full model and extracted the last token's
    // logits into shim_logits (correct even when output_all overrides all tokens to output).
    if (lbc->block_end == -1) {
        if (lbc->shim_logits.empty())
            throw std::runtime_error("lbc_project_to_logits (shim): no logits cached — call embedTokens first");
        return lbc->shim_logits;
    }

    llama_memory_seq_rm(llama_get_memory(lbc->ctx), 0, -1, -1);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]     = 0; // unused by llama_project_hidden_to_logits (hidden path)
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == n_tokens - 1) ? 1 : 0;
    }
    batch.n_tokens = n_tokens;

    int rc = llama_project_hidden_to_logits(lbc->ctx, batch, hidden);
    llama_batch_free(batch);
    if (rc != 0)
        throw std::runtime_error("llama_project_hidden_to_logits failed: rc=" + std::to_string(rc));

    // llama_get_logits returns a compact buffer with only the flagged tokens' logits.
    // We flagged exactly one token (the last), so logits[0..n_vocab) is the result.
    const float* logits = llama_get_logits(lbc->ctx);
    if (!logits) throw std::runtime_error("llama_get_logits returned null after projection");
    return std::vector<float>(logits, logits + n_vocab);
}

std::vector<float> lbc_forward(
        LlamaBlockContext* lbc, const float* hidden_in, int n_tokens) {
    if (n_tokens <= 0)
        throw std::runtime_error("lbc_forward: n_tokens must be > 0");
    const int32_t n_embd = llama_model_n_embd(lbc->model);
    std::vector<float> out((size_t)n_embd * n_tokens);

    // Stateless forward: clear residual KV state from previous calls on seq 0.
    llama_memory_seq_rm(llama_get_memory(lbc->ctx), 0, -1, -1);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]     = 0; // unused by llama_forward_hidden_range (hidden path)
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 0;
    }
    batch.n_tokens = n_tokens;

    int rc = llama_forward_hidden_range(
        lbc->ctx, batch, hidden_in, out.data(),
        lbc->block_start, lbc->block_end);
    llama_batch_free(batch);
    if (rc != 0)
        throw std::runtime_error("llama_forward_hidden_range failed: rc=" + std::to_string(rc));
    return out;
}

int lbc_session_open(LlamaBlockContext* lbc, int max_seq_len) {
    // Clamp to actual KV cache size so callers can't request more than n_ctx.
    const int n_ctx = (int)llama_n_ctx(lbc->ctx);
    if (max_seq_len > n_ctx) {
        fprintf(stderr, "lbc_session_open: clamping max_seq_len %d -> n_ctx %d\n", max_seq_len, n_ctx);
        max_seq_len = n_ctx;
    }

    // seq 0 is reserved for stateless calls; sessions use seq_ids 1..(n_seq_max-1).
    // n_seq_max is set to 64 at context creation, so max 63 concurrent sessions.
    static constexpr int MAX_SESSIONS = 63;
    if ((int)lbc->sessions.size() >= MAX_SESSIONS)
        throw std::runtime_error("lbc_session_open: maximum concurrent sessions (" +
                                 std::to_string(MAX_SESSIONS) + ") reached");

    // Allocate session_id: find one not already in use. Bounded loop (max MAX_SESSIONS
    // entries in map, so at most MAX_SESSIONS+1 iterations needed).
    int32_t sid = lbc->next_session_id++;
    if (lbc->next_session_id < 1) lbc->next_session_id = 1;
    for (int i = 0; i < MAX_SESSIONS && lbc->sessions.count(sid); ++i) {
        sid = lbc->next_session_id++;
        if (lbc->next_session_id < 1) lbc->next_session_id = 1;
    }
    if (lbc->sessions.count(sid))
        throw std::runtime_error("lbc_session_open: could not allocate a unique session_id");

    // Allocate seq_id: must not collide with any live session's KV-cache slot.
    // After int32 wraparound the counter can return a value still in use.
    auto seq_in_use = [&](int32_t s) {
        return std::any_of(lbc->sessions.begin(), lbc->sessions.end(),
            [s](const auto& kv){ return kv.second.seq_id == s; });
    };
    int32_t seq_id = lbc->next_seq_id++;
    if (lbc->next_seq_id < 1) lbc->next_seq_id = 1;
    for (int i = 0; i < MAX_SESSIONS && seq_in_use(seq_id); ++i) {
        seq_id = lbc->next_seq_id++;
        if (lbc->next_seq_id < 1) lbc->next_seq_id = 1;
    }
    if (seq_in_use(seq_id))
        throw std::runtime_error("lbc_session_open: could not allocate a unique seq_id");

    lbc->sessions[sid] = SessionInfo{ seq_id, 0, max_seq_len };
    return sid;
}

void lbc_session_close(LlamaBlockContext* lbc, int session_id) {
    auto it = lbc->sessions.find(session_id);
    if (it == lbc->sessions.end()) return;
    llama_memory_seq_rm(
        llama_get_memory(lbc->ctx),
        (llama_seq_id)it->second.seq_id,
        -1, -1);
    lbc->sessions.erase(it);
}

std::vector<float> lbc_session_forward(
        LlamaBlockContext* lbc, int session_id,
        const float* hidden_in, int n_new_tokens) {
    auto it = lbc->sessions.find(session_id);
    if (it == lbc->sessions.end())
        throw std::runtime_error("Invalid session id: " + std::to_string(session_id));
    if (it->second.n_past + n_new_tokens > it->second.max_length)
        throw std::runtime_error("lbc_session_forward: sequence exceeds max_length for session " + std::to_string(session_id));

    SessionInfo& info    = it->second;
    const int32_t n_embd = llama_model_n_embd(lbc->model);
    std::vector<float> out((size_t)n_embd * n_new_tokens);

    llama_batch batch = llama_batch_init(n_new_tokens, 0, 1);
    for (int i = 0; i < n_new_tokens; i++) {
        batch.pos[i]       = info.n_past + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = info.seq_id;
        batch.logits[i]    = 0;
    }
    batch.n_tokens = n_new_tokens;

    int rc = llama_forward_hidden_range(
        lbc->ctx, batch, hidden_in, out.data(),
        lbc->block_start, lbc->block_end);
    llama_batch_free(batch);
    if (rc != 0)
        throw std::runtime_error("lbc_session_forward decode failed: rc=" + std::to_string(rc));

    info.n_past += n_new_tokens;
    return out;
}

std::vector<float> lbc_session_decode_logits(
        LlamaBlockContext* lbc, int session_id,
        const int32_t* ids, int n_tokens) {
    if (n_tokens <= 0)
        throw std::runtime_error("lbc_session_decode_logits: n_tokens must be > 0");
    if (lbc->block_end != -1)
        throw std::runtime_error("lbc_session_decode_logits: only valid on shim contexts (block_end == -1)");
    auto it = lbc->sessions.find(session_id);
    if (it == lbc->sessions.end())
        throw std::runtime_error("lbc_session_decode_logits: invalid session id: " + std::to_string(session_id));
    if (it->second.n_past + n_tokens > it->second.max_length)
        throw std::runtime_error("lbc_session_decode_logits: sequence exceeds max_length for session " + std::to_string(session_id));

    SessionInfo& info = it->second;
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(lbc->model));

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]     = ids[i];
        batch.pos[i]       = info.n_past + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = info.seq_id;
        // Only flag logits on the last token; cp.embeddings=true forces output_all,
        // so the logit buffer will be n_tokens*n_vocab regardless.
        batch.logits[i]    = (i == n_tokens - 1) ? 1 : 0;
    }
    batch.n_tokens = n_tokens;

    int rc = llama_decode(lbc->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0)
        throw std::runtime_error("lbc_session_decode_logits: llama_decode failed rc=" + std::to_string(rc));

    info.n_past += n_tokens;

    // cp.embeddings=true forces output_all=true: all n_tokens positions get logits.
    // Last token's slice starts at (n_tokens-1)*n_vocab.
    const float* raw = llama_get_logits(lbc->ctx);
    if (!raw)
        throw std::runtime_error("lbc_session_decode_logits: llama_get_logits returned null");
    return std::vector<float>(
        raw + (size_t)(n_tokens - 1) * n_vocab,
        raw + (size_t)n_tokens * n_vocab
    );
}

std::vector<float> lbc_session_decode_logits_all(
        LlamaBlockContext* lbc, int session_id,
        const int32_t* ids, int n_tokens) {
    if (n_tokens <= 0)
        throw std::runtime_error("lbc_session_decode_logits_all: n_tokens must be > 0");
    if (lbc->block_end != -1)
        throw std::runtime_error("lbc_session_decode_logits_all: only valid on shim contexts (block_end == -1)");
    auto it = lbc->sessions.find(session_id);
    if (it == lbc->sessions.end())
        throw std::runtime_error("lbc_session_decode_logits_all: invalid session id: " + std::to_string(session_id));
    if (it->second.n_past + n_tokens > it->second.max_length)
        throw std::runtime_error("lbc_session_decode_logits_all: sequence exceeds max_length for session " + std::to_string(session_id));

    SessionInfo& info = it->second;
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(lbc->model));

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]     = ids[i];
        batch.pos[i]       = info.n_past + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = info.seq_id;
        batch.logits[i]    = 1; // Request logits for ALL positions
    }
    batch.n_tokens = n_tokens;

    int rc = llama_decode(lbc->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0)
        throw std::runtime_error("lbc_session_decode_logits_all: llama_decode failed rc=" + std::to_string(rc));

    info.n_past += n_tokens;

    const float* raw = llama_get_logits(lbc->ctx);
    if (!raw)
        throw std::runtime_error("lbc_session_decode_logits_all: llama_get_logits returned null");
    return std::vector<float>(raw, raw + (size_t)n_tokens * n_vocab);
}

void lbc_session_rollback(LlamaBlockContext* lbc, int session_id, int new_n_past) {
    auto it = lbc->sessions.find(session_id);
    if (it == lbc->sessions.end())
        throw std::runtime_error("lbc_session_rollback: invalid session id: " + std::to_string(session_id));

    SessionInfo& info = it->second;
    if (new_n_past < 0 || new_n_past > info.n_past)
        throw std::runtime_error("lbc_session_rollback: new_n_past " + std::to_string(new_n_past) +
                                 " out of range [0, " + std::to_string(info.n_past) + "]");

    if (new_n_past < info.n_past) {
        llama_memory_seq_rm(
            llama_get_memory(lbc->ctx),
            (llama_seq_id)info.seq_id,
            new_n_past, -1);
        info.n_past = new_n_past;
    }
}

