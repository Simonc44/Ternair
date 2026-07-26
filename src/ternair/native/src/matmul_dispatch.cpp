/**
 * matmul_dispatch.cpp — Runtime ISA dispatcher
 *
 * Selects the fastest available matmul kernel at runtime:
 *   AVX-512 > AVX-2 > scalar
 * Detection is done once at library load time via CPUID.
 */
#include "ternair_native.h"
#include <cstdint>
#include <cstring>

#if defined(_MSC_VER)
  #include <intrin.h>
  #define CPUID(regs, leaf) __cpuid((int*)(regs), (leaf))
#elif defined(__GNUC__) || defined(__clang__)
  #include <cpuid.h>
  static inline void CPUID(uint32_t regs[4], uint32_t leaf) {
      __cpuid_count(leaf, 0, regs[0], regs[1], regs[2], regs[3]);
  }
#else
  static inline void CPUID(uint32_t regs[4], uint32_t leaf) {
      memset(regs, 0, 16);
  }
#endif

namespace ternair {

enum class ISA { SCALAR, AVX2, AVX512 };

static ISA detect_isa() {
    uint32_t regs[4] = {0};
    CPUID(regs, 1);
    bool has_avx  = (regs[2] >> 28) & 1;
    CPUID(regs, 7);
    bool has_avx2   = (regs[1] >> 5)  & 1;
    bool has_avx512 = (regs[1] >> 16) & 1;
    if (has_avx && has_avx2 && has_avx512) return ISA::AVX512;
    if (has_avx && has_avx2)               return ISA::AVX2;
    return ISA::SCALAR;
}

static const ISA g_isa = detect_isa();

// Forward declarations
void matmul_scalar(const uint8_t*, const float*, const float*, float*, int, int);

// Weak symbols for optional SIMD backends (linked in if compiled)
__attribute__((weak))
void matmul_avx2(const uint8_t*, const float*, const float*, float*, int, int) {
    matmul_scalar(nullptr, nullptr, nullptr, nullptr, 0, 0);
}
__attribute__((weak))
void matmul_avx512(const uint8_t*, const float*, const float*, float*, int, int) {
    matmul_scalar(nullptr, nullptr, nullptr, nullptr, 0, 0);
}

void matmul_dispatch(
    const uint8_t* packed,
    const float*   x,
    const float*   gamma,
    float*         out,
    int out_f,
    int in_f
) {
    switch (g_isa) {
        case ISA::AVX512: matmul_avx512(packed, x, gamma, out, out_f, in_f); break;
        case ISA::AVX2:   matmul_avx2  (packed, x, gamma, out, out_f, in_f); break;
        default:          matmul_scalar(packed, x, gamma, out, out_f, in_f); break;
    }
}

const char* isa_name() {
    switch (g_isa) {
        case ISA::AVX512: return "avx512";
        case ISA::AVX2:   return "avx2";
        default:          return "scalar";
    }
}

} // namespace ternair
