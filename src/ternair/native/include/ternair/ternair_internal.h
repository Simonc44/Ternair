// ==========================================================================
// ternair_internal.h  —  Internal definitions shared by all .cpp files
// ==========================================================================
// NOT exposed to library consumers.  Defines the full struct of
// TernairRuntime + the thread pool helpers.  The public
// ternair/ternair_runtime.h keeps an opaque forward declaration so
// callers can't poke at the internals.
// ==========================================================================

#ifndef TERNAIR_INTERNAL_H_
#define TERNAIR_INTERNAL_H_

#include "ternair/ternair_runtime.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// ----------------------------------------------------------------------
// NativeModel: the on-disk weight + activation buffers we keep
// in-memory after ternair_load().  For v1.0 this is just the embedding
// table + the lm_head (a single ternary projection) + final norm.
// v1.1 grows this to per-layer weights.
// ----------------------------------------------------------------------

struct TernaryProjection {
    std::vector<uint8_t>  packed;    // (M, Kp) where M=vocab_size, Kp=ceil(hidden_size/4)
    std::vector<float>    gamma;     // (M,)
    int M = 0;
    int Kp = 0;
    int N = 0;  // hidden_size
};

struct NativeModel {
    std::vector<uint16_t> embed_table;       // vocab * hidden (fp16 raw bits)
    int vocab_size   = 0;
    int hidden_size  = 0;
    int max_seq_len  = 0;
    float rms_norm_eps = 1e-5f;

    TernaryProjection lm_head;
    std::vector<uint16_t> final_norm;        // hidden_size fp16
};

// ----------------------------------------------------------------------
// TernairRuntime: full definition (was previously defined in
// threadpool.cpp; moved here so that runtime.cpp + loader.cpp can
// access the fields without violating the opaque public API).
// ----------------------------------------------------------------------

struct TernairRuntime {
    // Thread pool
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex mu;
    std::condition_variable cv_task;
    std::condition_variable cv_done;
    std::atomic<bool> stop{false};
    int n_threads = 0;
    int n_active = 0;

    // Backend + config
    int backend = 0;     // TERNAIR_BACKEND_*
    int num_layers = 0;
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int vocab_size = 0;
    int max_seq_len = 0;
    float rms_norm_eps = 1e-5f;

    // Model buffers (opaque from the public header).
    NativeModel model;
};

// ----------------------------------------------------------------------
// Thread pool helpers
// ----------------------------------------------------------------------

// Initialise the worker pool.  num_threads <= 0 picks
// std::thread::hardware_concurrency().  Idempotent.
void ternair_pool_init(TernairRuntime* rt, int num_threads);

// Submit N tasks (each callable takes a 0-based stripe index).
// Blocks until all tasks complete.
void ternair_pool_submit(TernairRuntime* rt, int n_tasks,
                          const std::function<void(int)>& body);

#endif  // TERNAIR_INTERNAL_H_
