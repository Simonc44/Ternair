/**
 * matmul_scalar.cpp — Scalar fallback for ternary matmul
 *
 * Pure C++ implementation, no SIMD intrinsics.
 * Used when AVX-2 / AVX-512 are unavailable.
 */
#include "ternair_native.h"
#include <cstring>
#include <cstdint>
#include <cmath>

namespace ternair {

// Decode 2-bit packed trits: 0→0, 1→+1, 2→-1, 3→0 (padding)
static inline int8_t decode_trit(uint8_t bits) {
    switch (bits & 0x3) {
        case 1:  return  1;
        case 2:  return -1;
        default: return  0;
    }
}

void matmul_scalar(
    const uint8_t* __restrict__ packed,   // (out_f, in_f/4) uint8
    const float*   __restrict__ x,        // (in_f,)         fp32
    const float*   __restrict__ gamma,    // (out_f,)        fp32 scale
    float*         __restrict__ out,      // (out_f,)        fp32 output
    int out_f,
    int in_f
) {
    const int k_packed = in_f / 4;
    for (int o = 0; o < out_f; ++o) {
        float acc = 0.0f;
        const uint8_t* row = packed + o * k_packed;
        int i = 0;
        // Unroll 4 bytes at a time
        for (; i + 4 <= k_packed; i += 4) {
            uint8_t b0 = row[i+0], b1 = row[i+1], b2 = row[i+2], b3 = row[i+3];
            int base = (i) * 4;
            // byte 0
            acc += decode_trit(b0 >> 0) * x[base+0];
            acc += decode_trit(b0 >> 2) * x[base+1];
            acc += decode_trit(b0 >> 4) * x[base+2];
            acc += decode_trit(b0 >> 6) * x[base+3];
            // byte 1
            acc += decode_trit(b1 >> 0) * x[base+4];
            acc += decode_trit(b1 >> 2) * x[base+5];
            acc += decode_trit(b1 >> 4) * x[base+6];
            acc += decode_trit(b1 >> 6) * x[base+7];
            // byte 2
            acc += decode_trit(b2 >> 0) * x[base+8];
            acc += decode_trit(b2 >> 2) * x[base+9];
            acc += decode_trit(b2 >> 4) * x[base+10];
            acc += decode_trit(b2 >> 6) * x[base+11];
            // byte 3
            acc += decode_trit(b3 >> 0) * x[base+12];
            acc += decode_trit(b3 >> 2) * x[base+13];
            acc += decode_trit(b3 >> 4) * x[base+14];
            acc += decode_trit(b3 >> 6) * x[base+15];
        }
        // Remainder
        for (; i < k_packed; ++i) {
            uint8_t b = row[i];
            int base = i * 4;
            acc += decode_trit(b >> 0) * x[base+0];
            acc += decode_trit(b >> 2) * x[base+1];
            acc += decode_trit(b >> 4) * x[base+2];
            acc += decode_trit(b >> 6) * x[base+3];
        }
        out[o] = acc * gamma[o];
    }
}

} // namespace ternair
