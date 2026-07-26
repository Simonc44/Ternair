/**
 * runtime.cpp — Higher-level batched matmul + FP16<->FP32 conversion
 */
#include "ternair_native.h"
#include <cstdint>
#include <cstring>
#include <cmath>

namespace ternair {

extern void matmul_dispatch(
    const uint8_t*, const float*, const float*, float*, int, int);

} // namespace ternair

extern "C" {

// FP16 bit pattern to float (software, no intrinsics)
static float fp16_to_float(uint16_t h) {
    uint32_t sign     = (h >> 15) & 1;
    uint32_t exp      = (h >> 10) & 0x1F;
    uint32_t mantissa =  h        & 0x3FF;
    if (exp == 0x1F) {
        // inf or nan
        uint32_t bits = (sign << 31) | 0x7F800000 | (mantissa << 13);
        float f; memcpy(&f, &bits, 4); return f;
    }
    if (exp == 0) {
        // subnormal
        float f = (float)mantissa / 1024.0f;
        return sign ? -f : f;
    }
    uint32_t bits = (sign << 31) | ((exp + 112) << 23) | (mantissa << 13);
    float f; memcpy(&f, &bits, 4);
    return f;
}

static uint16_t float_to_fp16(float f) {
    uint32_t bits; memcpy(&bits, &f, 4);
    uint16_t sign     = (bits >> 31) & 1;
    int32_t  exp      = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFF;
    if (exp <= 0)  return sign << 15;
    if (exp >= 31) return (sign << 15) | 0x7C00;
    return (sign << 15) | (exp << 10) | (mantissa >> 13);
}

/**
 * ternair_matmul_fp16
 *
 * Same as ternair_matmul but accepts/returns FP16 (uint16_t bit patterns).
 * Used by ternair.native._native_forward() to avoid a round-trip through
 * Python for the fp32<->fp16 conversion.
 */
void ternair_matmul_fp16(
    const uint8_t*  packed,    // (out_f, in_f/4) uint8
    const uint16_t* x_fp16,   // (in_f,)          fp16 bits
    const float*    gamma,     // (out_f,)         fp32
    uint16_t*       out_fp16,  // (out_f,)         fp16 bits (output)
    int out_f,
    int in_f
) {
    // Convert x from fp16 -> fp32
    float* x_fp32 = new float[in_f];
    for (int i = 0; i < in_f; ++i)
        x_fp32[i] = fp16_to_float(x_fp16[i]);

    // Run the dispatch matmul
    float* out_fp32 = new float[out_f];
    ternair::matmul_dispatch(packed, x_fp32, gamma, out_fp32, out_f, in_f);

    // Convert output fp32 -> fp16
    for (int i = 0; i < out_f; ++i)
        out_fp16[i] = float_to_fp16(out_fp32[i]);

    delete[] x_fp32;
    delete[] out_fp32;
}

} // extern "C"
