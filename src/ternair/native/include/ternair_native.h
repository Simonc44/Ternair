/**
 * ternair_native.h — Internal header for the native C++ engine
 */
#pragma once
#include <cstdint>
#include <functional>

namespace ternair {

// Scalar fallback (always available)
void matmul_scalar(
    const uint8_t* packed,
    const float*   x,
    const float*   gamma,
    float*         out,
    int out_f,
    int in_f
);

// Runtime ISA dispatcher (picks avx512 > avx2 > scalar)
void matmul_dispatch(
    const uint8_t* packed,
    const float*   x,
    const float*   gamma,
    float*         out,
    int out_f,
    int in_f
);

// Norm
void rms_norm(
    const float* x,
    const float* weight,
    float*       out,
    int n,
    float eps = 1e-6f
);

void layer_norm(
    const float* x,
    const float* weight,
    const float* bias,
    float*       out,
    int n,
    float eps = 1e-5f
);

// Thread pool
void threadpool_init(int n_threads = 0);
void parallel_for(int n, std::function<void(int)> fn);
int  threadpool_n_threads();

} // namespace ternair
