#include <napi.h>
#include <unordered_map>
#include <atomic>
#include <vector>
#include "model_context.h"
#include "block_forward.h"

static std::unordered_map<uint32_t, ModelContext*> g_handles;
static std::atomic<uint32_t> g_next_handle{1};

// ── hello ─────────────────────────────────────────────────────────────────────
Napi::String Hello(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), "coral-native ready");
}

// ── loadBlockRange ────────────────────────────────────────────────────────────
Napi::Number LoadBlockRange(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 4 || !info[0].IsString() ||
        !info[1].IsNumber() || !info[2].IsNumber() || !info[3].IsNumber()) {
        Napi::TypeError::New(env,
            "loadBlockRange(modelPath: string, blockStart: number, blockEnd: number, totalBlocks: number)"
        ).ThrowAsJavaScriptException();
        return {};
    }
    std::string path   = info[0].As<Napi::String>();
    int block_start    = info[1].As<Napi::Number>().Int32Value();
    int block_end      = info[2].As<Napi::Number>().Int32Value();
    int total_blocks   = info[3].As<Napi::Number>().Int32Value();
    try {
        ModelContext* mc = model_context_load(path.c_str(), block_start, block_end, total_blocks);
        if (!mc) { Napi::Error::New(env, "model_context_load returned null").ThrowAsJavaScriptException(); return {}; }
        uint32_t handle = g_next_handle.fetch_add(1);
        g_handles[handle] = mc;
        return Napi::Number::New(env, handle);
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return {};
    }
}

// ── freeBlockRange ────────────────────────────────────────────────────────────
Napi::Value FreeBlockRange(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsNumber()) { return env.Undefined(); }
    uint32_t handle = info[0].As<Napi::Number>().Uint32Value();
    auto it = g_handles.find(handle);
    if (it != g_handles.end()) { model_context_free(it->second); g_handles.erase(it); }
    return env.Undefined();
}

// ── runForward ────────────────────────────────────────────────────────────────
Napi::Value RunForward(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3 || !info[0].IsNumber() ||
        !info[1].IsTypedArray() || !info[2].IsNumber()) {
        Napi::TypeError::New(env,
            "runForward(handle: number, input: Float32Array, nTokens: number)"
        ).ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uint32_t handle  = info[0].As<Napi::Number>().Uint32Value();
    int      n_tokens = info[2].As<Napi::Number>().Int32Value();
    auto it = g_handles.find(handle);
    if (it == g_handles.end()) {
        Napi::Error::New(env, "Invalid handle").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    auto ta = info[1].As<Napi::TypedArray>();
    if (ta.TypedArrayType() != napi_float32_array) {
        Napi::TypeError::New(env, "Input must be Float32Array").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    const float* input_ptr = ta.As<Napi::Float32Array>().Data();
    try {
        std::vector<float> output = block_runner_forward(it->second, input_ptr, n_tokens);
        Napi::Float32Array result = Napi::Float32Array::New(env, output.size());
        if (!output.empty()) {
            memcpy(result.Data(), output.data(), output.size() * sizeof(float));
        }
        return result;
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}

// ── module registration ───────────────────────────────────────────────────────
Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set("hello",          Napi::Function::New(env, Hello));
    exports.Set("loadBlockRange", Napi::Function::New(env, LoadBlockRange));
    exports.Set("freeBlockRange", Napi::Function::New(env, FreeBlockRange));
    exports.Set("runForward",     Napi::Function::New(env, RunForward));
    return exports;
}

NODE_API_MODULE(coral_native, Init)
