#pragma once
#include "model_context.h"
#include <vector>

// Run hidden states through loaded transformer blocks.
// input_data: float32 values, length = n_tokens × config.n_embd
// Returns float32 output, same length.
std::vector<float> block_runner_forward(
    ModelContext* mc,
    const float*  input_data,
    int           n_tokens
);
