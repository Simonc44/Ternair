// ==========================================================================
// matmul_avx2.cpp  —  AVX2 + F16C ternary matmul
// ==========================================================================
// Universal x86-64 fallback.  Uses:
//   - _mm256_cvtph_ps   (F16C)   to convert fp16 -> fp32
//   - 4 x __m256 accumulators (16 fp32 each) per output row block of 16
//   - bit manipulation to extract trit signs from packed bytes
//
// For each output row m, each kp loads 4 consecutive x values (kp*4,
// kp*4+1, kp*4+2, kp*4+3) and applies the 4 trit signs.
//
// 16 output rows are processed in parallel: for each kp, the 16 packed
// bytes give 4 vectors of 16 trit signs (one per trit position p), each
// broadcast against the same 4 x values.
// ==========================================================================

#include "ternair/ternair_runtime.h"

#if !defined(__AVX__) || !defined(__F16C__)
#  error "matmul_avx2.cpp requires AVX + F16C"
#endif

#include <immintrin.h>
#include <cstring>
#include <vector>

namespace {

// IEEE 754 binary16 -> float32 using F16C.
inline float fp16_to_f32_f16c(uint16_t h) noexcept {
    __m128i v = _mm_set1_epi16(static_cast<short>(h));
    __m128 f  = _mm_cvtph_ps(v);
    float out;
    _mm_store_ss(&out, f);
    return out;
}

}  // namespace


extern "C" int ternair_ternary_matmul_avx2(
    const uint8_t* packed, int M, int Kp,
    const uint16_t* x_fp16_bits, int N,
    const float* gamma,
    uint16_t* out_fp16_bits
) {
    if (M <= 0 || Kp <= 0 || N <= 0) return 1;

    // Decode x once (scalar) for simplicity; F16C gather would help
    // marginally and adds complexity -- skipped for v1.0.
    std::vector<float> x_f32(static_cast<size_t>(N));
    for (int n = 0; n < N; ++n) {
        x_f32[n] = fp16_to_f32_f16c(x_fp16_bits[n]);
    }

    // Process 16 output rows in parallel via four __m256 accumulators
    // (one per trit position across 4 trits/byte = 16 trits/4 = 4 vecs).
    //
    // The 4 trit positions in a packed byte share 4 consecutive x
    // values, so we keep 4 separate accumulators and fold at the end.
    const int ROWS_PER_VEC = 16;

    for (int m_block = 0; m_block < M; m_block += ROWS_PER_VEC) {
        int m_count = std::min(ROWS_PER_VEC, M - m_block);
        // 16 x FP32 accumulators, one per (row, trit-position)
        __m256 acc[16] = { _mm256_setzero_ps() };

        for (int kp = 0; kp < Kp; ++kp) {
            // Load up to 16 packed bytes (one per output row in the block)
            uint8_t buf[16] = {0};
            for (int i = 0; i < m_count; ++i) {
                buf[i] = packed[(m_block + i) * Kp + kp];
            }
            __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(buf));

            for (int p = 0; p < 4; ++p) {
                // Round-trip through 16-bit to avoid _mm_srli_epi16 cross-byte
                // contamination.  (No per-byte shift intrinsic until AVX-VBMI2.)
                __m256i bytes16 = _mm256_cvtepu8_epi16(bytes);
                __m256i shifted = _mm256_srli_epi16(bytes16, 2 * p);
                __m256i masked  = _mm256_and_si256(shifted,
                                                   _mm256_set1_epi16(0x0003));
                // SSE2-safe 16-i16 -> 16-i8 truncation (no AVX-512VL/BW required).
                __m128i two_bit = _mm_packus_epi16(
                    _mm256_castsi256_si128(masked),
                    _mm256_extracti128_si256(masked, 1));

                // pos = low bit of two_bit (no shift, safe).
                __m128i pos_byte = _mm_and_si128(two_bit, _mm_set1_epi8(0x01));
                // neg = (two_bit >> 1) & 0x01 -- also round-trip for safety.
                __m256i tb16     = _mm256_cvtepu8_epi16(two_bit);
                __m256i neg_sh16 = _mm256_srli_epi16(tb16, 1);
                __m256i neg_and  = _mm256_and_si256(neg_sh16,
                                                   _mm256_set1_epi16(0x0001));
                __m128i neg_byte = _mm_packus_epi16(
                    _mm256_castsi256_si128(neg_and),
                    _mm256_extracti128_si256(neg_and, 1));

                __m128i trit_i8  = _mm_sub_epi8(pos_byte, neg_byte);  // {-1, 0, +1}

                // Sign-extend int8 -> int32, then float.
                __m256i trit_i32 = _mm256_cvtepi8_epi32(trit_i8);
                __m256  trit_f32 = _mm256_cvtepi32_ps(trit_i32);

                // Load x[kp*4+p] (one fp32 value, broadcast to 16)
                int n = kp * 4 + p;
                if (n < N) {
                    __m256 x_b = _mm256_set1_ps(x_f32[n]);
                    // acc[row] += trit_f32[row] * x
                    for (int r = 0; r < m_count; ++r) {
                        // Per-row multiply via broadcast + fma; we use
                        // _mm256_permutevar_ps to broadcast the r-th element.
                        // Simpler: scalar extract + broadcast.
                        float trit_r = _mm256_cvtss_f32(_mm256_permutevar_ps(
                            trit_f32, _mm256_set1_epi32(r)));
                        acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(trit_r), x_b, acc[r]);
                    }
                }
            }
        }

        // Multiply by gamma and store (fp32 -> fp16 via scalar cast).
        // acc[r] is a __m256 with the same scalar broadcast to all 8 lanes
        // (the inner loop runs over rows but only scalar accumulation per
        // row).  Just extract lane 0 -- no horizontal sum needed, which
        // would otherwise multiply the result by 8.
        for (int r = 0; r < m_count; ++r) {
            float total = _mm256_cvtss_f32(acc[r]);
            float result = total * gamma[m_block + r];
            // fp32 -> fp16
            __m128i r_i = _mm_cvtps_ph(_mm_set_ss(result), 0);
            uint16_t h = static_cast<uint16_t>(_mm_extract_epi16(r_i, 0));
            out_fp16_bits[m_block + r] = h;
        }
    }
    return 0;
}
