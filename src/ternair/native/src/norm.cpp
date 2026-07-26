// ==========================================================================
// norm.cpp  —  RMSNorm implementation (FP32 accumulator)
// ==========================================================================

#include "ternair/ternair_runtime.h"

#include <cmath>
#include <cstring>

namespace {
inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t fbits;
    if (exp == 0) fbits = sign;
    else if (exp == 31) fbits = sign | (0xFFu << 23) | (mant << 13);
    else fbits = sign | ((exp + 112u) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &fbits, sizeof(f));
    return f;
}
inline uint16_t f32_to_fp16(float f) {
    uint32_t fbits;
    std::memcpy(&fbits, &f, sizeof(fbits));
    uint32_t sign = (fbits >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((fbits >> 23) & 0xFF) - 112;
    uint32_t mant = (fbits >> 13) & 0x3FFu;
    if (exp <= 0)  return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7BFFu);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}
}

extern "C" void ternair_rmsnorm(
    uint16_t* x_fp16,         // in-place (n,)
    int n,
    const uint16_t* weight_fp16,  // (n,)
    float eps
) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; ++i) {
        float v = fp16_to_f32(x_fp16[i]);
        sum_sq += v * v;
    }
    float rms = 1.0f / std::sqrt(sum_sq / n + eps);
    for (int i = 0; i < n; ++i) {
        float v = fp16_to_f32(x_fp16[i]) * rms * fp16_to_f32(weight_fp16[i]);
        x_fp16[i] = f32_to_fp16(v);
    }
}
