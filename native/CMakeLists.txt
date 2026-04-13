cmake_minimum_required(VERSION 3.20)
project(coral_native CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Fetch llama.cpp (includes ggml and gguf) ──────────────────────────────────
include(FetchContent)
FetchContent_Declare(
  llama
  GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
  GIT_TAG        b5300
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
)

# Disable llama.cpp build targets we don't need
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama)

# ── node-addon-api include path ───────────────────────────────────────────────
execute_process(
  COMMAND node -e "process.stdout.write(require('path').resolve(require('node-addon-api').include_dir))"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..
  OUTPUT_VARIABLE NAPI_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "NAPI include dir: ${NAPI_INCLUDE_DIR}")

# ── coral_native shared library (.node) ──────────────────────────────────────
add_library(coral_native SHARED
  src/coral_napi.cpp
  src/model_context.cpp
  src/block_forward.cpp
  ${CMAKE_JS_SRC}
)

target_include_directories(coral_native PRIVATE
  ${CMAKE_JS_INC}
  ${NAPI_INCLUDE_DIR}
  ${llama_SOURCE_DIR}
  ${llama_SOURCE_DIR}/include
  ${llama_SOURCE_DIR}/ggml/include
  src
)

target_link_libraries(coral_native PRIVATE
  llama
  ggml
  ${CMAKE_JS_LIB}
)

target_compile_definitions(coral_native PRIVATE
  NAPI_VERSION=8
  NODE_ADDON_API_DISABLE_DEPRECATED
)

# .node extension, no lib prefix
set_target_properties(coral_native PROPERTIES
  PREFIX  ""
  SUFFIX  ".node"
)
