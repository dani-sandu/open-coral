#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "llama.h"

struct VocabContext {
    llama_model* model = nullptr;
};

struct VocabSpecialTokens {
    int32_t bos_id    = 0;
    int32_t eos_id    = 0;
    int32_t eot_id    = -1;
    int32_t vocab_size = 0;
};

VocabContext*        vocab_context_load(const char* path);
void                 vocab_context_free(VocabContext* vc);
std::vector<int32_t> vocab_tokenize(VocabContext* vc, const std::string& text, bool add_special, bool parse_special);
std::string          vocab_token_to_piece(VocabContext* vc, int32_t token_id);
std::string          vocab_apply_chat_template(VocabContext* vc, const std::string& user_message);
VocabSpecialTokens   vocab_get_special_tokens(VocabContext* vc);
