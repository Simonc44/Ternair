/**
 * loader.cpp — Library init / teardown
 */
#include "ternair_native.h"
#include <cstdio>

namespace ternair {

extern const char* isa_name();
extern void threadpool_init(int);
extern int  threadpool_n_threads();

static bool g_initialised = false;

void library_init(int n_threads) {
    if (g_initialised) return;
    threadpool_init(n_threads);
    g_initialised = true;
}

void library_destroy() {
    g_initialised = false;
}

const char* library_isa() {
    return isa_name();
}

int library_n_threads() {
    return threadpool_n_threads();
}

} // namespace ternair

// C ABI exported symbols (resolved by ternair.native.__init__.py)
extern "C" {

void ternair_init(int n_threads) {
    ternair::library_init(n_threads);
}

void ternair_destroy() {
    ternair::library_destroy();
}

const char* ternair_isa() {
    return ternair::library_isa();
}

int ternair_n_threads() {
    return ternair::library_n_threads();
}

// Main C entry point: ternary matmul for one output row batch
// packed : (out_f, in_f/4) uint8
// x      : (in_f,)          fp32
// gamma  : (out_f,)         fp32
// out    : (out_f,)         fp32
void ternair_matmul(
    const uint8_t* packed,
    const float*   x,
    const float*   gamma,
    float*         out,
    int out_f,
    int in_f
) {
    ternair::matmul_dispatch(packed, x, gamma, out, out_f, in_f);
}

} // extern "C"
