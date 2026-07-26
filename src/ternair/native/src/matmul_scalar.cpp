// ==========================================================================
// matmul_scalar.cpp  —  Reference ternary matmul (always available)
// ==========================================================================
// Pure scalar implementation, no SIMD.  Correct but slow.  Used as the
// final fallback and as the reference for the SIMD implementations to
// match.
// ==========================================================================

#include "ternair/ternair_runtime.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace {

// Decode the 4 trits packed in one byte (4 trits/byte, 2-bit each).
// bits = (byte >> (2*pos)) & 3   -> trit = (bits & 1) - ((bits >> 1) & 1)
inline void decode_byte(uint8_t byte, int8_t t[4]) noexcept {
    for (int p = 0; p < 4; ++p) {
        int bits = (byte >> (2 * p)) & 3;
        t[p] = static_cast<int8_t>((bits & 1) - ((bits >> 1) & 1));
    }
}

// IEEE 754 binary16 -> float32 (no intrinsics).
inline float fp16_to_f32(uint16_t h) noexcept {
    uint32_t sign = (static_cast<uint32_t>(h) >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t fbits;
    if (exp == 0) {
        // Subnormal or zero -> flush to zero
        fbits = sign;
    } else if (exp == 31) {
        // Inf / NaN
        fbits = sign | (0xFFu << 23) | (mant << 13);
    } else {
        fbits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &fbits, sizeof(f));
    return f;
}

// float32 -> IEEE 754 binary16 (round-to-nearest-even on the mantissa).
inline uint16_t f32_to_fp16(float f) noexcept {
    uint32_t fbits;
    std::memcpy(&fbits, &f, sizeof(fbits));
    uint32_t sign = (fbits >> 16) & 0x8000u;
    int32_t  exp  = static_cast<int32_t>((fbits >> 23) & 0xFF) - 112;
    uint32_t mant = (fbits >> 13) & 0x3FFu;
    if (exp <= 0)    return static_cast<uint16_t>(sign);  // flush subnormals
    if (exp >= 31)   return static_cast<uint16_t>(sign | 0x7BFFu);  // +/-inf
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
}

}  // namespace


extern "C" int ternair_ternary_matmul_scalar(
    const uint8_t* packed, int M, int Kp,
    const uint16_t* x_fp16_bits, int N,
    const float* gamma,
    uint16_t* out_fp16_bits
) {
    if (M <= 0 || Kp <= 0 || N <= 0) return 1;

    // Decode x once (avoid repeating the binary16 -> float32 work inside
    // the hot loop).
    std::vector<float> x_f32(static_cast<size_t>(N));
    for (int n = 0; n < N; ++n) {
        x_f32[n] = fp16_to_f32(x_fp16_bits[n]);
    }

    for (int m = 0; m < M; ++m) {
        float acc = 0.0f;
        const uint8_t* row = packed + static_cast<size_t>(m) * Kp;
        for (int kp = 0; kp < Kp; ++kp) {
            int8_t t[4];
            decode_byte(row[kp], t);
            for (int p = 0; p < 4; ++p) {
                int n = kp * 4 + p;
                if (n < N && t[p] != 0) {
                    acc += static_cast<float>(t[p]) * x_f32[n];
                }
            }
        }
        out_fp16_bits[m] = f32_to_fp16(acc * gamma[m]);
    }
    return 0;
}
