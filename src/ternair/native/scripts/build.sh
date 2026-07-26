#!/usr/bin/env bash
# ==========================================================================
# scripts/build.sh  —  Build the Ternair native C++ engine (g++-only)
# ==========================================================================
# No CMake required: uses g++ directly with auto-detected SIMD ISA.
#
# Usage:
#   bash scripts/build.sh           # build into ./build/
#   bash scripts/build.sh clean     # wipe ./build/
# ==========================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BUILD_DIR="$ROOT/build"

if [[ "${1:-}" == "clean" ]]; then
    rm -rf "$BUILD_DIR"
    echo "[build.sh] cleaned $BUILD_DIR"
    exit 0
fi

mkdir -p "$BUILD_DIR"

CXX="${CXX:-g++}"
CXXFLAGS_BASE="-std=c++17 -O3 -fPIC -fvisibility=hidden -Wall -Wextra -DNDEBUG"
# Shared library needs default visibility so ctypes (Python) can resolve
# every ternair_* entry point.  Executables (CLI + test) stay hidden since
# they statically link all symbols.
CXXFLAGS_LIB="-std=c++17 -O3 -fPIC -fvisibility=default -Wall -Wextra -DNDEBUG"
INC="-I$ROOT/include"

# Auto-detect AVX-512F+AVX-512BW+AVX-512DQ
CPUFLAGS=""
if [[ "$(uname -m)" == "x86_64" ]]; then
    if grep -q -E '^flags.*(avx512f|avx512dq)' /proc/cpuinfo 2>/dev/null; then
        CPUFLAGS="-mavx512f -mavx512bw -mavx512dq -mavx2 -mf16c -mfma"
        echo "[build.sh] AVX-512F+BW+DQ ENABLED"
    elif grep -q -E '^flags.*avx2' /proc/cpuinfo 2>/dev/null; then
        CPUFLAGS="-mavx2 -mf16c -mfma"
        echo "[build.sh] AVX2 + F16C + FMA ENABLED"
    else
        echo "[build.sh] SCALAR only (no AVX2 detected)"
    fi
else
    echo "[build.sh] non-x86_64 host -- SCALAR only"
fi

# Sources (matmul_avx512.cpp requires AVX-512; matmul_avx2.cpp requires AVX2+F16C)
LIB_SOURCES=(
    "$ROOT/src/matmul_scalar.cpp"
    "$ROOT/src/matmul_dispatch.cpp"
    "$ROOT/src/threadpool.cpp"
    "$ROOT/src/loader.cpp"
    "$ROOT/src/norm.cpp"
    "$ROOT/src/runtime.cpp"
)
if [[ "$CPUFLAGS" == *avx512f* ]]; then
    LIB_SOURCES+=("$ROOT/src/matmul_avx512.cpp")
fi
if [[ "$CPUFLAGS" == *avx2* ]]; then
    # AVX-512 implies AVX2+F16C, so compile both when AVX-512 is enabled.
    LIB_SOURCES+=("$ROOT/src/matmul_avx2.cpp")
fi

# Library
echo "[build.sh] compiling libternair_native.so..."
"$CXX" $CXXFLAGS_LIB $CPUFLAGS $INC -shared -o "$BUILD_DIR/libternair_native.so" \
    "${LIB_SOURCES[@]}" -lpthread

# CLI
echo "[build.sh] compiling ternair_native_cli..."
CLI_SOURCES=("${LIB_SOURCES[@]}" "$ROOT/src/main.cpp")
"$CXX" $CXXFLAGS_LIB $CPUFLAGS $INC -o "$BUILD_DIR/ternair_native_cli" \
    "${CLI_SOURCES[@]}" -L"$BUILD_DIR" -lternair_native -lpthread

# C++ smoke test
echo "[build.sh] compiling ternair_native_test..."
TEST_SOURCES=("$ROOT/src/matmul_scalar.cpp" "$ROOT/src/matmul_dispatch.cpp" \
              "$ROOT/tests/test_matmul.cpp")
if [[ "$CPUFLAGS" == *avx512f* ]]; then
    TEST_SOURCES+=("$ROOT/src/matmul_avx512.cpp")
fi
if [[ "$CPUFLAGS" == *avx2* ]]; then
    TEST_SOURCES+=("$ROOT/src/matmul_avx2.cpp")
fi
"$CXX" $CXXFLAGS_LIB $CPUFLAGS $INC -o "$BUILD_DIR/ternair_native_test" \
    "${TEST_SOURCES[@]}" -L"$BUILD_DIR" -lternair_native -lpthread

echo
echo "=== Run C++ smoke test ==="
"$BUILD_DIR/ternair_native_test"

echo
echo "=== Run Python integration test ==="
if command -v python3 >/dev/null; then
    PYTHONPATH="$ROOT/python:src" LD_LIBRARY_PATH="$BUILD_DIR" \
        python3 "$ROOT/tests/test_native.py" || true
fi

echo
echo "[build.sh] artefacts in $BUILD_DIR:"
ls -la "$BUILD_DIR"/libternair_native.so "$BUILD_DIR"/ternair_native_cli "$BUILD_DIR"/ternair_native_test 2>&1
