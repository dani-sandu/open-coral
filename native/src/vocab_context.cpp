#include "vocab_context.h"
#include <stdexcept>

VocabContext* vocab_context_load(const char* path) {
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;

    llama_model* model = llama_model_load_from_file(path, params);
    if (!model) {
        throw std::runtime_error(std::string("vocab_context_load: failed to load ") + path);
    }

    VocabContext* vc = new VocabContext();
    vc->model = model;
    return vc;
}

void vocab_context_free(VocabContext* vc) {
    if (!vc) return;
    if (vc->model) llama_model_free(vc->model);
    delete vc;
}

std::vector<int32_t> vocab_tokenize(VocabContext* vc, const std::string& text, bool add_special, bool parse_special) {
    const llama_vocab* vocab = llama_model_get_vocab(vc->model);

    // First pass: determine how many tokens we'll get
    int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                               nullptr, 0, add_special, parse_special);
    if (n == 0) return {};
    int32_t capacity = -n;

    std::vector<int32_t> tokens(capacity);
    int32_t actual = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                                    tokens.data(), capacity, add_special, parse_special);
    if (actual < 0) {
        throw std::runtime_error("vocab_tokenize: buffer too small after two-pass sizing");
    }
    tokens.resize(actual);
    return tokens;
}

std::string vocab_token_to_piece(VocabContext* vc, int32_t token_id) {
    const llama_vocab* vocab = llama_model_get_vocab(vc->model);
    char buf[256];
    int32_t n = llama_token_to_piece(vocab, token_id, buf, sizeof(buf), 0, false);
    if (n == 0) return "";
    if (n < 0) {
        // Buffer too small — allocate exactly what's needed and retry
        std::string result(-n, '\0');
        int32_t n2 = llama_token_to_piece(vocab, token_id, result.data(), -n, 0, false);
        if (n2 <= 0) return "";
        result.resize(n2);
        return result;
    }
    return std::string(buf, n);
}

std::string vocab_apply_chat_template(VocabContext* vc, const std::string& user_message) {
    const char* tmpl = llama_model_chat_template(vc->model, nullptr);
    if (!tmpl) {
        // Model has no embedded chat template — return raw message
        return user_message;
    }

    llama_chat_message msg = { "user", user_message.c_str() };

    // First pass: measure required buffer size
    int32_t n = llama_chat_apply_template(tmpl, &msg, (size_t)1, true, nullptr, 0);
    if (n < 0) {
        throw std::runtime_error("vocab_apply_chat_template: template application failed");
    }

    std::string result(n, '\0');
    int32_t written = llama_chat_apply_template(tmpl, &msg, (size_t)1, true, result.data(), n);
    if (written < 0) {
        throw std::runtime_error("vocab_apply_chat_template: template write failed");
    }
    result.resize(written);
    return result;
}

std::string vocab_apply_chat_template_multi(VocabContext* vc, const std::vector<ChatTurn>& turns) {
    const char* tmpl = llama_model_chat_template(vc->model, nullptr);
    if (!tmpl) {
        // The substring "no embedded chat template" is the sentinel detected by
        // src/inference/native-tokenizer.ts::encodeChatMulti to throw a typed
        // ChatTemplateUnavailableError in JS. Keep these in sync.
        throw std::runtime_error("vocab_apply_chat_template_multi: model has no embedded chat template");
    }

    if (turns.empty()) {
        throw std::runtime_error("vocab_apply_chat_template_multi: turns must not be empty");
    }

    std::vector<llama_chat_message> msgs;
    msgs.reserve(turns.size());
    for (const auto& t : turns) {
        msgs.push_back({ t.role.c_str(), t.content.c_str() });
    }

    int32_t n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true, nullptr, 0);
    if (n < 0) {
        throw std::runtime_error("vocab_apply_chat_template_multi: template application failed");
    }

    std::string result((size_t)n, '\0');
    int32_t written = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true, result.data(), n);
    if (written < 0) {
        throw std::runtime_error("vocab_apply_chat_template_multi: template write failed");
    }
    result.resize((size_t)written);
    return result;
}

VocabSpecialTokens vocab_get_special_tokens(VocabContext* vc) {
    const llama_vocab* vocab = llama_model_get_vocab(vc->model);
    VocabSpecialTokens st;
    st.bos_id    = llama_vocab_bos(vocab);
    st.eos_id    = llama_vocab_eos(vocab);
    st.eot_id    = llama_vocab_eot(vocab);
    st.vocab_size = llama_vocab_n_tokens(vocab);
    return st;
}
