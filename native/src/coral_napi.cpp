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
    return Napi::String::New(info.Env(), "opencoral-native ready");
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

// ── loadBlockRangeSharded ─────────────────────────────────────────────────────
// Args: shardPaths: string[], blockStart: number, blockEnd: number, totalBlocks: number
Napi::Number LoadBlockRangeSharded(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 4 || !info[0].IsArray() ||
        !info[1].IsNumber() || !info[2].IsNumber() || !info[3].IsNumber()) {
        Napi::TypeError::New(env,
            "loadBlockRangeSharded(shardPaths: string[], blockStart, blockEnd, totalBlocks)"
        ).ThrowAsJavaScriptException();
        return {};
    }

    auto arr = info[0].As<Napi::Array>();
    std::vector<std::string> paths_storage;
    std::vector<const char*> paths;
    paths_storage.reserve(arr.Length());
    paths.reserve(arr.Length());

    for (uint32_t i = 0; i < arr.Length(); i++) {
        Napi::Value v = arr[i];
        if (!v.IsString()) {
            Napi::TypeError::New(env, "shardPaths must be an array of strings").ThrowAsJavaScriptException();
            return {};
        }
        paths_storage.push_back(v.As<Napi::String>().Utf8Value());
        paths.push_back(paths_storage.back().c_str());
    }

    int block_start  = info[1].As<Napi::Number>().Int32Value();
    int block_end    = info[2].As<Napi::Number>().Int32Value();
    int total_blocks = info[3].As<Napi::Number>().Int32Value();

    try {
        ModelContext* mc = model_context_load_shards(
            paths.data(), (int)paths.size(), block_start, block_end, total_blocks
        );
        if (!mc) { Napi::Error::New(env, "model_context_load_shards returned null").ThrowAsJavaScriptException(); return {}; }
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

// ── embedTokens ───────────────────────────────────────────────────────────────
Napi::Value EmbedTokens(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsTypedArray()) {
        Napi::TypeError::New(env,
            "embedTokens(handle: number, tokenIds: Int32Array)"
        ).ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uint32_t handle = info[0].As<Napi::Number>().Uint32Value();
    auto it = g_handles.find(handle);
    if (it == g_handles.end()) {
        Napi::Error::New(env, "Invalid handle").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    auto ta = info[1].As<Napi::TypedArray>();
    if (ta.TypedArrayType() != napi_int32_array) {
        Napi::TypeError::New(env, "tokenIds must be Int32Array").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    const int32_t* ids = ta.As<Napi::Int32Array>().Data();
    int n_tokens = (int)ta.ElementLength();
    try {
        std::vector<float> output = embed_tokens(it->second, ids, n_tokens);
        Napi::Float32Array result = Napi::Float32Array::New(env, output.size());
        if (!output.empty()) memcpy(result.Data(), output.data(), output.size() * sizeof(float));
        return result;
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}

// ── projectToLogits ───────────────────────────────────────────────────────────
Napi::Value ProjectToLogits(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3 || !info[0].IsNumber() ||
        !info[1].IsTypedArray() || !info[2].IsNumber()) {
        Napi::TypeError::New(env,
            "projectToLogits(handle: number, hidden: Float32Array, nTokens: number)"
        ).ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uint32_t handle = info[0].As<Napi::Number>().Uint32Value();
    int n_tokens = info[2].As<Napi::Number>().Int32Value();
    auto it = g_handles.find(handle);
    if (it == g_handles.end()) {
        Napi::Error::New(env, "Invalid handle").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    auto ta = info[1].As<Napi::TypedArray>();
    if (ta.TypedArrayType() != napi_float32_array) {
        Napi::TypeError::New(env, "hidden must be Float32Array").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    const float* hidden_ptr = ta.As<Napi::Float32Array>().Data();
    try {
        std::vector<float> output = project_to_logits(it->second, hidden_ptr, n_tokens);
        Napi::Float32Array result = Napi::Float32Array::New(env, output.size());
        if (!output.empty()) memcpy(result.Data(), output.data(), output.size() * sizeof(float));
        return result;
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}

// ── getVocabSize ──────────────────────────────────────────────────────────────
Napi::Value GetVocabSize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsNumber()) {
        Napi::TypeError::New(env, "getVocabSize(handle: number)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uint32_t handle = info[0].As<Napi::Number>().Uint32Value();
    auto it = g_handles.find(handle);
    if (it == g_handles.end()) {
        Napi::Error::New(env, "Invalid handle").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    return Napi::Number::New(env, it->second->config.n_vocab);
}

// ── openSession ───────────────────────────────────────────────────────────────
Napi::Value OpenSession(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsNumber()) {
        Napi::TypeError::New(env, "openSession(handle, maxLength)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uint32_t handle = info[0].As<Napi::Number>().Uint32Value();
    int maxLen      = info[1].As<Napi::Number>().Int32Value();
    auto it = g_handles.find(handle);
    if (it == g_handles.end()) {
        Napi::Error::New(env, "Invalid handle").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    try {
        int sid = create_session(it->second, maxLen);
        return Napi::Number::New(env, sid);
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}

// ── closeSession ──────────────────────────────────────────────────────────────
Napi::Value CloseSession(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsNumber()) {
        return env.Undefined();
    }
    uint32_t handle = info[0].As<Napi::Number>().Uint32Value();
    int sid         = info[1].As<Napi::Number>().Int32Value();
    auto it = g_handles.find(handle);
    if (it != g_handles.end()) free_session(it->second, sid);
    return env.Undefined();
}

// ── sessionForward ────────────────────────────────────────────────────────────
Napi::Value SessionForward(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 4 || !info[0].IsNumber() || !info[1].IsNumber() ||
        !info[2].IsTypedArray() || !info[3].IsNumber()) {
        Napi::TypeError::New(env,
            "sessionForward(handle, sessionId, input: Float32Array, nNewTokens)"
        ).ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uint32_t handle  = info[0].As<Napi::Number>().Uint32Value();
    int sid          = info[1].As<Napi::Number>().Int32Value();
    int nNew         = info[3].As<Napi::Number>().Int32Value();
    auto it = g_handles.find(handle);
    if (it == g_handles.end()) {
        Napi::Error::New(env, "Invalid handle").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    auto ta = info[2].As<Napi::TypedArray>();
    if (ta.TypedArrayType() != napi_float32_array) {
        Napi::TypeError::New(env, "Input must be Float32Array").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    const float* input_ptr = ta.As<Napi::Float32Array>().Data();
    try {
        std::vector<float> output = session_forward(it->second, sid, input_ptr, nNew);
        Napi::Float32Array result = Napi::Float32Array::New(env, output.size());
        if (!output.empty()) memcpy(result.Data(), output.data(), output.size() * sizeof(float));
        return result;
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}

// ── module registration ───────────────────────────────────────────────────────
Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set("hello",                 Napi::Function::New(env, Hello));
    exports.Set("loadBlockRange",        Napi::Function::New(env, LoadBlockRange));
    exports.Set("loadBlockRangeSharded", Napi::Function::New(env, LoadBlockRangeSharded));
    exports.Set("freeBlockRange",        Napi::Function::New(env, FreeBlockRange));
    exports.Set("runForward",       Napi::Function::New(env, RunForward));
    exports.Set("embedTokens",     Napi::Function::New(env, EmbedTokens));
    exports.Set("projectToLogits", Napi::Function::New(env, ProjectToLogits));
    exports.Set("getVocabSize",    Napi::Function::New(env, GetVocabSize));
    exports.Set("openSession",     Napi::Function::New(env, OpenSession));
    exports.Set("closeSession",    Napi::Function::New(env, CloseSession));
    exports.Set("sessionForward",  Napi::Function::New(env, SessionForward));
    return exports;
}

NODE_API_MODULE(coral_native, Init)
