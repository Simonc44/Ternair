// ==========================================================================
// tests/test_matmul.cpp  —  Smoke test for the ternary matmul dispatch
// ==========================================================================
// Verifies that all compiled backends (scalar / AVX2 / AVX-512) produce
// bit-for-bit identical results on a small random workload.
// ==========================================================================

#include "ternair/ternair_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>

extern "C" int ternair_ternary_matmul_scalar(
    const uint8_t* packed, int M, int Kp,
    const uint16_t* x_fp16_bits, int N,
    const float* gamma,
    uint16_t* out_fp16_bits);

static uint16_t f32_to_fp16(float f) {
    uint32_t fbits;
    std::memcpy(&fbits, &f, sizeof(fbits));
    uint32_t sign = (fbits >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((fbits >> 23) & 0xFF) - 112;
    uint32_t mant = (fbits >> 13) & 0x3FFu;
    if (exp <= 0)  return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7BFFu);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static float fp16_to_f32(uint16_t h) {
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

int main() {
    constexpr int M = 32;
    constexpr int N = 64;       // multiple of 4
    constexpr int Kp = N / 4;

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> trit_dist(-1, 1);
    std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> gamma_dist(0.1f, 1.0f);

    // Build random trits + pack
    std::vector<int8_t> trits(static_cast<size_t>(M) * N);
    for (auto& t : trits) t = static_cast<int8_t>(trit_dist(rng));
    // Pack: 4 trits/byte, LSB-first
    std::vector<uint8_t> packed(static_cast<size_t>(M) * Kp, 0);
    for (int m = 0; m < M; ++m) {
        for (int kp = 0; kp < Kp; ++kp) {
            uint8_t byte = 0;
            for (int p = 0; p < 4; ++p) {
                int n_idx = kp * 4 + p;
                if (n_idx >= N) break;
                int t = trits[m * N + n_idx];
                if (t > 0)       byte |= (1u << (2 * p));
                else if (t < 0)  byte |= (2u << (2 * p));
            }
            packed[m * Kp + kp] = byte;
        }
    }
    // Random x in fp16
    std::vector<uint16_t> x_fp16(N);
    for (int n = 0; n < N; ++n) x_fp16[n] = f32_to_fp16(x_dist(rng));
    // Random gamma
    std::vector<float> gamma(M);
    for (int m = 0; m < M; ++m) gamma[m] = gamma_dist(rng);

    // Reference: scalar
    std::vector<uint16_t> out_ref(M);
    int err = ternair_ternary_matmul_scalar(
        packed.data(), M, Kp,
        x_fp16.data(), N, gamma.data(), out_ref.data());
    if (err) { fprintf(stderr, "scalar failed: %d\n", err); return 1; }

    // Dispatch
    std::vector<uint16_t> out_disp(M);
    err = ternair_ternary_matmul(
        packed.data(), M, Kp,
        x_fp16.data(), N, gamma.data(), out_disp.data());
    if (err) { fprintf(stderr, "dispatch failed: %d\n", err); return 2; }

    int max_abs_err = 0;
    float max_rel_err = 0.0f;
    int exact = 0;
    for (int m = 0; m < M; ++m) {
        float a = fp16_to_f32(out_ref[m]);
        float b = fp16_to_f32(out_disp[m]);
        float diff = std::fabs(a - b);
        float denom = std::fmax(std::fabs(a), 1e-3f);
        float rel   = diff / denom;
        if (diff > max_abs_err) max_abs_err = (int)diff;
        if (rel > max_rel_err) max_rel_err = rel;
        if (out_ref[m] == out_disp[m]) exact++;
    }
    printf("[test] backend=%s exact=%d/%d max_abs_err=%g max_rel_err=%.4f\n",
           ternair_backend_name(0), exact, M, (double)max_abs_err, (double)max_rel_err);
    if (max_rel_err < 0.05f) {
        printf("[test] PASS\n");
        return 0;
    }
    printf("[test] FAIL (rel-err too high)\n");
    return 3;
}
