# apply-once.cmake — PATCH_COMMAND for FetchContent (llama.cpp b8233).
# Adds open-coral block-range execution support:
#   - block_range_start / block_range_end in llama_model_params
#   - llama_get_hidden_from_tokens()
#   - llama_project_hidden_to_logits()
#   - llama_forward_hidden_range()
#
# Uses string(REPLACE) on the fetched source so that the modifications survive
# minor upstream whitespace changes better than git-format patches.

set(SENTINEL "${BINARY_DIR}/.coral_patches_applied")
if(EXISTS "${SENTINEL}")
  message(STATUS "[coral] Patches already applied — skipping.")
  return()
endif()

message(STATUS "[coral] Applying open-coral block-range patches to llama.cpp ...")

# ─── helper: read, replace, write ─────────────────────────────────────────────
macro(coral_replace FILE OLD NEW)
  file(READ "${SOURCE_DIR}/${FILE}" _content)
  string(FIND "${_content}" "${OLD}" _pos)
  if(_pos EQUAL -1)
    message(FATAL_ERROR "[coral] Anchor not found in ${FILE}:\n---\n${OLD}\n---")
  endif()
  string(REPLACE "${OLD}" "${NEW}" _content "${_content}")
  file(WRITE "${SOURCE_DIR}/${FILE}" "${_content}")
endmacro()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. include/llama.h — add fields to llama_model_params
# ═══════════════════════════════════════════════════════════════════════════════

coral_replace("include/llama.h"
  "        bool no_alloc;        // only load metadata and simulate memory allocations
    };"
  "        bool no_alloc;        // only load metadata and simulate memory allocations

        // [open-coral] block-range execution for model sharding
        int32_t block_range_start; // first transformer block (inclusive), -1 = all
        int32_t block_range_end;   // last transformer block (inclusive), -1 = all
    };"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. include/llama.h — add 3 API function declarations after llama_decode
# ═══════════════════════════════════════════════════════════════════════════════

coral_replace("include/llama.h"
  "    // Set the number of threads used for decoding
    // n_threads is the number of threads used for generation (single token)
    // n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
    LLAMA_API void llama_set_n_threads(struct llama_context * ctx, int32_t n_threads, int32_t n_threads_batch);"
  "    // [open-coral] Token IDs -> hidden states (post-norm).
    // Runs full decode with embeddings=true, copies n_embd*n_tokens floats to out.
    LLAMA_API int32_t llama_get_hidden_from_tokens(
            struct llama_context * ctx,
              struct llama_batch   batch,
                           float * out);

    // [open-coral] Hidden states -> logits. Runs output_norm + lm_head only (no layers).
    LLAMA_API int32_t llama_project_hidden_to_logits(
            struct llama_context * ctx,
              struct llama_batch   batch,
                     const float * hidden);

    // [open-coral] Hidden states -> hidden states through layers [start..end] (inclusive).
    LLAMA_API int32_t llama_forward_hidden_range(
            struct llama_context * ctx,
              struct llama_batch   batch,
                     const float * hidden_in,
                           float * hidden_out,
                         int32_t   block_start,
                         int32_t   block_end);

    // Set the number of threads used for decoding
    // n_threads is the number of threads used for generation (single token)
    // n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
    LLAMA_API void llama_set_n_threads(struct llama_context * ctx, int32_t n_threads, int32_t n_threads_batch);"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. src/llama-model.cpp — add defaults in llama_model_default_params()
# ═══════════════════════════════════════════════════════════════════════════════

coral_replace("src/llama-model.cpp"
  "        /*.no_alloc                    =*/ false,
    };"
  "        /*.no_alloc                    =*/ false,
        /*.block_range_start            =*/ -1,
        /*.block_range_end              =*/ -1,
    };"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. src/llama-graph.h — add fields to llm_graph_params
# ═══════════════════════════════════════════════════════════════════════════════

coral_replace("src/llama-graph.h"
  "    uint32_t n_outputs;

    llm_graph_cb cb;"
  "    uint32_t n_outputs;

    // [open-coral] block-range graph options
    int32_t coral_block_start = -1;  // first layer (inclusive), -1 = 0
    int32_t coral_block_end   = -1;  // last layer (inclusive), -1 = n_layer-1
    bool    coral_skip_norm   = false;
    bool    coral_skip_lm_head = false;

    llm_graph_cb cb;"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. src/llama-graph.h — add fields to llm_graph_context
# ═══════════════════════════════════════════════════════════════════════════════

coral_replace("src/llama-graph.h"
  "    const int32_t n_ctx_orig; // yarn

    const enum llama_pooling_type pooling_type;"
  "    const int32_t n_ctx_orig; // yarn

    // [open-coral]
    const int32_t coral_block_start;
    const int32_t coral_block_end;
    const bool    coral_skip_norm;
    const bool    coral_skip_lm_head;

    const enum llama_pooling_type pooling_type;"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. src/llama-graph.h — add to allow_reuse()
# ═══════════════════════════════════════════════════════════════════════════════

coral_replace("src/llama-graph.h"
  "            cross == other.cross;
    }"
  "            cross == other.cross &&
            coral_block_start  == other.coral_block_start &&
            coral_block_end    == other.coral_block_end &&
            coral_skip_norm    == other.coral_skip_norm &&
            coral_skip_lm_head == other.coral_skip_lm_head;
    }"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 7. src/llama-graph.cpp — initialize new fields in llm_graph_context ctor
# ═══════════════════════════════════════════════════════════════════════════════

coral_replace("src/llama-graph.cpp"
  "    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    pooling_type     (cparams.pooling_type),"
  "    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    coral_block_start(params.coral_block_start),
    coral_block_end  (params.coral_block_end),
    coral_skip_norm  (params.coral_skip_norm),
    coral_skip_lm_head(params.coral_skip_lm_head),
    pooling_type     (cparams.pooling_type),"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 8. src/llama-context.h — add coral_graph_options + method declarations
# ═══════════════════════════════════════════════════════════════════════════════

coral_replace("src/llama-context.h"
  "    int decode(const llama_batch & batch_inp);"
  "    int decode(const llama_batch & batch_inp);

    // [open-coral] block-range API
    int get_hidden_from_tokens(const llama_batch & batch_inp, float * out);
    int project_hidden_to_logits(const llama_batch & batch_inp, const float * hidden);
    int forward_hidden_range(const llama_batch & batch_inp, const float * hidden_in, float * hidden_out, int32_t bstart, int32_t bend);"
)

coral_replace("src/llama-context.h"
  "    bool has_evaluated_once = false;"
  "    // [open-coral] per-call graph overrides (reset after each API call)
    struct coral_graph_options {
        int32_t block_start  = -1;
        int32_t block_end    = -1;
        bool    skip_norm    = false;
        bool    skip_lm_head = false;
    } coral_opts_;

    bool has_evaluated_once = false;"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 9. src/llama-context.cpp — add coral fields to graph_params()
# ═══════════════════════════════════════════════════════════════════════════════

coral_replace("src/llama-context.cpp"
  "        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,"
  "        /*.n_outputs   =*/ n_outputs,
        /*.coral_block_start  =*/ coral_opts_.block_start,
        /*.coral_block_end    =*/ coral_opts_.block_end,
        /*.coral_skip_norm    =*/ coral_opts_.skip_norm,
        /*.coral_skip_lm_head =*/ coral_opts_.skip_lm_head,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 10. src/llama-context.cpp — append implementations
# ═══════════════════════════════════════════════════════════════════════════════

file(READ "${SOURCE_DIR}/src/llama-context.cpp" _ctx_cpp)
string(APPEND _ctx_cpp "
// ══════════════════════════════════════════════════════════════════════════════
// [open-coral] block-range execution API
// ══════════════════════════════════════════════════════════════════════════════

int llama_context::get_hidden_from_tokens(const llama_batch & batch_inp, float * out) {
    // Thin wrapper: decode() already produces embeddings when cparams.embeddings == true.
    int rc = decode(batch_inp);
    if (rc != 0) return rc;

    float * embd_ptr = get_embeddings();
    if (!embd_ptr) return -1;

    const int64_t n_embd_out = model.hparams.n_embd_out();
    std::memcpy(out, embd_ptr, (size_t)n_embd_out * batch_inp.n_tokens * sizeof(float));
    return 0;
}

int llama_context::project_hidden_to_logits(const llama_batch & batch_inp, const float * hidden) {
    // Build a batch that feeds hidden states as float embeddings.
    llama_batch embd_batch = batch_inp;
    embd_batch.token = nullptr;
    embd_batch.embd  = const_cast<float *>(hidden);

    // Skip all transformer layers, keep norm + lm_head.
    const auto saved = coral_opts_;
    coral_opts_.block_start  = (int32_t)model.hparams.n_layer; // il_start >= il_end => no layers
    coral_opts_.block_end    = (int32_t)model.hparams.n_layer - 1;
    coral_opts_.skip_norm    = false;
    coral_opts_.skip_lm_head = false;

    int rc = decode(embd_batch);
    coral_opts_ = saved;
    return rc;
}

int llama_context::forward_hidden_range(
        const llama_batch & batch_inp, const float * hidden_in, float * hidden_out,
        int32_t bstart, int32_t bend) {
    llama_batch embd_batch = batch_inp;
    embd_batch.token = nullptr;
    embd_batch.embd  = const_cast<float *>(hidden_in);

    const auto saved = coral_opts_;
    coral_opts_.block_start  = bstart;
    coral_opts_.block_end    = bend;
    coral_opts_.skip_norm    = true;
    coral_opts_.skip_lm_head = true;

    int rc = decode(embd_batch);
    coral_opts_ = saved;

    if (rc != 0) return rc;

    float * embd_ptr = get_embeddings();
    if (!embd_ptr) return -1;

    const int64_t n_embd_out = model.hparams.n_embd_out();
    std::memcpy(hidden_out, embd_ptr, (size_t)n_embd_out * batch_inp.n_tokens * sizeof(float));
    return 0;
}

// ── public C wrappers ────────────────────────────────────────────────────────

int32_t llama_get_hidden_from_tokens(llama_context * ctx, llama_batch batch, float * out) {
    const int ret = ctx->get_hidden_from_tokens(batch, out);
    if (ret != 0) {
        LLAMA_LOG_ERROR(\"%s: failed, ret = %d\\n\", __func__, ret);
    }
    return ret;
}

int32_t llama_project_hidden_to_logits(llama_context * ctx, llama_batch batch, const float * hidden) {
    const int ret = ctx->project_hidden_to_logits(batch, hidden);
    if (ret != 0) {
        LLAMA_LOG_ERROR(\"%s: failed, ret = %d\\n\", __func__, ret);
    }
    return ret;
}

int32_t llama_forward_hidden_range(
        llama_context * ctx, llama_batch batch,
        const float * hidden_in, float * hidden_out,
        int32_t block_start, int32_t block_end) {
    const int ret = ctx->forward_hidden_range(batch, hidden_in, hidden_out, block_start, block_end);
    if (ret != 0) {
        LLAMA_LOG_ERROR(\"%s: failed, ret = %d\\n\", __func__, ret);
    }
    return ret;
}
")
file(WRITE "${SOURCE_DIR}/src/llama-context.cpp" "${_ctx_cpp}")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. src/models/llama.cpp — modify layer loop to use block_range
# ═══════════════════════════════════════════════════════════════════════════════

# 11a. Replace the layer loop start
coral_replace("src/models/llama.cpp"
  "    for (int il = 0; il < n_layer; ++il) {"
  "    const int il_start = (coral_block_start >= 0) ? coral_block_start : 0;
    const int il_end   = (coral_block_end   >= 0) ? (coral_block_end + 1) : (int)n_layer;

    for (int il = il_start; il < il_end; ++il) {"
)

# 11b. Fix the output-id filtering (was n_layer - 1, now il_end - 1)
coral_replace("src/models/llama.cpp"
  "        if (il == n_layer - 1 && inp_out_ids) {"
  "        if (il == il_end - 1 && inp_out_ids) {"
)

# 11c. Wrap norm + lm_head in coral conditionals
coral_replace("src/models/llama.cpp"
  "    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, \"result_norm\", -1);
    res->t_embd = cur;

    if constexpr (!embed) {
        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, \"result_output\", -1);
        res->t_logits = cur;
    }"
  "    if (!coral_skip_norm) {
        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, \"result_norm\", -1);
    }
    res->t_embd = cur;

    if constexpr (!embed) {
        if (!coral_skip_lm_head) {
            // lm_head
            cur = build_lora_mm(model.output, cur);

            cb(cur, \"result_output\", -1);
            res->t_logits = cur;
        }
    }"
)

# ═══════════════════════════════════════════════════════════════════════════════
# Done — write sentinel
# ═══════════════════════════════════════════════════════════════════════════════

file(WRITE "${SENTINEL}" "applied")
message(STATUS "[coral] All patches applied successfully.")
