// ==========================================================================
// ternair_runtime.h  —  Public C API for the Ternair native runtime
// ==========================================================================
//
// Pure C, no external dependencies. Loads a SafeTensors model with ternary
// (2-bit fastpacked) weights, runs an end-to-end forward pass using
// AVX2 / AVX-512 SIMD intrinsics, and exposes a simple generate() entry
// point for the CLI + Python ctypes wrapper.
//
// Build modes (auto-detected by the build system):
//   * AVX-512 F32  (AVX512F + AVX512BW + AVX512DQ) -- default on x86-64
//   * AVX2 + F16C  (always on x86-64 since Haswell) -- fallback 1
//   * Scalar                                          -- fallback 2
//
// The packed weight format is 4 trits / byte.  Trit encoding:
//   bits = (byte >> (2*pos)) & 3
//   trit = (bits & 1) - ((bits >> 1) & 1)   // {-1, 0, +1}
//
// All activations are FP16; the accumulator is FP32 for numerical headroom.
// ==========================================================================

#ifndef TERNAIR_RUNTIME_H_
#define TERNAIR_RUNTIME_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle.  Allocated by ternair_create(), released by ternair_free().
typedef struct TernairRuntime TernairRuntime;

// Backend tags reported by ternair_backend_name().
#define TERNAIR_BACKEND_SCALAR  0
#define TERNAIR_BACKEND_AVX2     1
#define TERNAIR_BACKEND_AVX512   2

// ----------------------------------------------------------------------
// Lifecycle
// ----------------------------------------------------------------------

// Allocate a new (empty) runtime.  Returns NULL on OOM.
TernairRuntime* ternair_create(void);

// Release a runtime and all its allocated buffers.
void ternair_free(TernairRuntime* rt);

// Load a SafeTensors file.  ``num_threads`` <= 0 means auto-detect.
// Returns 0 on success, non-zero on failure.
int ternair_load(
    TernairRuntime* rt,
    const char* safetensors_path,
    int num_threads
);

// Report which SIMD backend the runtime selected at load time.
int ternair_backend(const TernairRuntime* rt);

// Human-readable backend name ("scalar" / "avx2" / "avx512").
const char* ternair_backend_name(int backend);

// Report the model config (read-only views).
int     ternair_num_layers(const TernairRuntime* rt);
int     ternair_hidden_size(const TernairRuntime* rt);
int     ternair_intermediate_size(const TernairRuntime* rt);
int     ternair_num_attention_heads(const TernairRuntime* rt);
int     ternair_num_kv_heads(const TernairRuntime* rt);
int     ternair_vocab_size(const TernairRuntime* rt);
int     ternair_max_seq_len(const TernairRuntime* rt);

// ----------------------------------------------------------------------
// Forward / generate
// ----------------------------------------------------------------------

// Compute logits[vocab] for the last token of ``input_ids`` (length
// ``input_len``).  ``logits_out`` must point to at least vocab_size floats.
// Returns 0 on success.
int ternair_forward(
    TernairRuntime* rt,
    const int32_t* input_ids,
    int input_len,
    float* logits_out
);

// Greedy / sampling-free generation.  ``prompt`` is an array of
// ``prompt_len`` token ids.  ``out_ids`` must be at least
// prompt_len + max_new_tokens.  Stops at ``eos_token_id`` if >= 0.
// Sampling parameters (T, top_k, top_p, repetition_penalty) are
// passed through; a future build may add a JSON config for full
// llama.cpp parity.  Returns the number of tokens written.
int ternair_generate(
    TernairRuntime* rt,
    const int32_t* prompt,
    int prompt_len,
    int32_t* out_ids,
    int max_new_tokens,
    int eos_token_id,
    float temperature,
    int top_k,
    float top_p,
    float repetition_penalty
);

// Standalone ternary matmul exported for benchmarking / Python tests.
// ``packed`` is row-major (M, Kp) uint8.  ``x_fp16`` is (N,) fp16 raw
// bits interpreted as IEEE 754 binary16.  ``gamma`` is (M,) fp32.
// ``out_fp16`` is (M,) fp16 storage.  Returns 0 on success.
int ternair_ternary_matmul(
    const uint8_t* packed,
    int M, int Kp,
    const uint16_t* x_fp16_bits,
    int N,
    const float* gamma,
    uint16_t* out_fp16_bits
);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TERNAIR_RUNTIME_H_
