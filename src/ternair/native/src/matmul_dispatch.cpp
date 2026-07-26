// ==========================================================================
// matmul_dispatch.cpp  —  CPU feature detection + dispatch
// ==========================================================================
// At load time, pick the best ternary matmul backend available on the
// host.  Each backend implements the same signature; the dispatcher
// routes ternair_ternary_matmul() to the right one.
// ==========================================================================

#include "ternair/ternair_runtime.h"

#include <cstring>

extern "C" int ternair_ternary_matmul_scalar(
    const uint8_t* packed, int M, int Kp,
    const uint16_t* x_fp16_bits, int N,
    const float* gamma,
    uint16_t* out_fp16_bits);

#if defined(__AVX__) && defined(__F16C__)
extern "C" int ternair_ternary_matmul_avx2(
    const uint8_t* packed, int M, int Kp,
    const uint16_t* x_fp16_bits, int N,
    const float* gamma,
    uint16_t* out_fp16_bits);
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
extern "C" int ternair_ternary_matmul_avx512(
    const uint8_t* packed, int M, int Kp,
    const uint16_t* x_fp16_bits, int N,
    const float* gamma,
    uint16_t* out_fp16_bits);
#endif


// ---------------------------------------------------------------------------
// CPU feature detection (x86-64)
// ---------------------------------------------------------------------------

#if defined(__x86_64__) || defined(_M_X64)
#  if defined(_MSC_VER)
#    include <intrin.h>
#  else
#    include <cpuid.h>
#  endif

static void cpuid(int info[4], int leaf, int sub) {
#  if defined(_MSC_VER)
    __cpuidex(info, leaf, sub);
#  else
    __cpuid_count(leaf, sub, info[0], info[1], info[2], info[3]);
#  endif
}

static int has_avx512f() {
    int info[4];
    cpuid(info, 7, 0);
    // Bit 16 of EBX = AVX-512F
    return (info[1] >> 16) & 1;
}
static int has_avx2() {
    int info[4];
    cpuid(info, 7, 0);
    // Bit 5 of EBX = AVX2
    return (info[1] >> 5) & 1;
}
static int has_f16c() {
    int info[4];
    cpuid(info, 1, 0);
    // Bit 29 of ECX = F16C
    return (info[2] >> 29) & 1;
}
#endif


// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

extern "C" int ternair_ternary_matmul(
    const uint8_t* packed, int M, int Kp,
    const uint16_t* x_fp16_bits, int N,
    const float* gamma,
    uint16_t* out_fp16_bits
) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
    if (has_avx512f()) {
        return ternair_ternary_matmul_avx512(
            packed, M, Kp, x_fp16_bits, N, gamma, out_fp16_bits);
    }
#endif
#if defined(__AVX__) && defined(__F16C__)
    if (has_avx2() && has_f16c()) {
        return ternair_ternary_matmul_avx2(
            packed, M, Kp, x_fp16_bits, N, gamma, out_fp16_bits);
    }
#endif
    return ternair_ternary_matmul_scalar(
        packed, M, Kp, x_fp16_bits, N, gamma, out_fp16_bits);
}


extern "C" int ternair_backend_name_(int backend) {
    switch (backend) {
        case TERNAIR_BACKEND_AVX512: return 2;
        case TERNAIR_BACKEND_AVX2:    return 1;
        case TERNAIR_BACKEND_SCALAR:
        default:                       return 0;
    }
}

extern "C" const char* ternair_backend_name(int backend) {
    switch (backend) {
        case TERNAIR_BACKEND_AVX512: return "avx512";
        case TERNAIR_BACKEND_AVX2:    return "avx2";
        case TERNAIR_BACKEND_SCALAR:
        default:                       return "scalar";
    }
}
