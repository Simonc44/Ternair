/**
 * norm.cpp — RMSNorm and LayerNorm in C++
 */
#include "ternair_native.h"
#include <cmath>
#include <cstdint>

namespace ternair {

void rms_norm(
    const float* __restrict__ x,      // (n,)
    const float* __restrict__ weight,  // (n,)
    float*       __restrict__ out,     // (n,)
    int n,
    float eps
) {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    float rms = 1.0f / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * rms * weight[i];
}

void layer_norm(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float*       __restrict__ out,
    int n,
    float eps
) {
    float mean = 0.0f;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= n;
    float var = 0.0f;
    for (int i = 0; i < n; ++i) { float d = x[i]-mean; var += d*d; }
    var /= n;
    float inv_std = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < n; ++i) {
        out[i] = (x[i] - mean) * inv_std * weight[i];
        if (bias) out[i] += bias[i];
    }
}

} // namespace ternair
