# ==========================================================================
# CMakeLists.txt  —  Build the Ternair native C++ engine
# ==========================================================================
# Usage:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   cmake --build . -j
#   ctest --output-on-failure
# ==========================================================================

cmake_minimum_required(VERSION 3.16)
project(ternair_native CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Source list
# ---------------------------------------------------------------------------
set(TERNAIR_NATIVE_SOURCES
    src/matmul_scalar.cpp
    src/matmul_dispatch.cpp
    src/threadpool.cpp
    src/loader.cpp
    src/norm.cpp
    src/runtime.cpp
    src/main.cpp
)

set(TERNAIR_NATIVE_LIB_SOURCES
    src/matmul_scalar.cpp
    src/matmul_dispatch.cpp
    src/threadpool.cpp
    src/loader.cpp
    src/norm.cpp
    src/runtime.cpp
)

# ---------------------------------------------------------------------------
# Backend detection: prefer AVX-512, fall back to AVX2+F16C, else scalar
# ---------------------------------------------------------------------------
include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

set(TERNAIR_HAS_AVX512 OFF)
set(TERNAIR_HAS_AVX2   OFF)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
    set(_x86 TRUE)
else()
    set(_x86 FALSE)
endif()

# Try AVX-512
if(_x86)
    set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512bw -mavx512dq")
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main() { __m512 a = _mm512_setzero_ps(); (void)a; return 0; }
    " _avx512_ok)
    if(_avx512_ok)
        set(TERNAIR_HAS_AVX512 ON)
        add_compile_options(-mavx512f -mavx512bw -mavx512dq)
        message(STATUS "[ternair_native] AVX-512F + AVX-512BW + AVX-512DQ ENABLED")
    else()
        message(STATUS "[ternair_native] AVX-512 NOT available")
    endif()

    # Try AVX2 + F16C
    set(CMAKE_REQUIRED_FLAGS "-mavx2 -mf16c -mfma")
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main() { __m256 a = _mm256_setzero_ps(); (void)a; return 0; }
    " _avx2_ok)
    if(_avx2_ok)
        set(TERNAIR_HAS_AVX2 ON)
        add_compile_options(-mavx2 -mf16c -mfma)
        message(STATUS "[ternair_native] AVX2 + F16C + FMA ENABLED")
    endif()
endif()

# Common optimisation flags (only when not MSVC)
if(NOT MSVC)
    add_compile_options(-O3 -fno-plt -fvisibility=hidden)
    if(_x86)
        add_compile_options(-mtune=generic)
    endif()
endif()

# ---------------------------------------------------------------------------
# Shared library: libternair_native.so
# ---------------------------------------------------------------------------
add_library(ternair_native SHARED ${TERNAIR_NATIVE_LIB_SOURCES})
target_include_directories(ternair_native PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_definitions(ternair_native PRIVATE
    $<$<BOOL:${TERNAIR_HAS_AVX512}>:TERNAIR_HAS_AVX512>
    $<$<BOOL:${TERNAIR_HAS_AVX2}>:TERNAIR_HAS_AVX2>
)
set_target_properties(ternair_native PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
    VERSION 0.6.0
    SOVERSION 0
)

if(TERNAIR_HAS_AVX512)
    target_sources(ternair_native PRIVATE src/matmul_avx512.cpp)
    target_compile_options(ternair_native PRIVATE -mavx512f -mavx512bw -mavx512dq)
endif()
if(TERNAIR_HAS_AVX2 AND NOT TERNAIR_HAS_AVX512)
    target_sources(ternair_native PRIVATE src/matmul_avx2.cpp)
    target_compile_options(ternair_native PRIVATE -mavx2 -mf16c -mfma)
endif()

# ---------------------------------------------------------------------------
# CLI binary
# ---------------------------------------------------------------------------
add_executable(ternair_native_cli ${TERNAIR_NATIVE_SOURCES})
target_link_libraries(ternair_native_cli PRIVATE ternair_native)
target_compile_definitions(ternair_native_cli PRIVATE
    $<$<BOOL:${TERNAIR_HAS_AVX512}>:TERNAIR_HAS_AVX512>
    $<$<BOOL:${TERNAIR_HAS_AVX2}>:TERNAIR_HAS_AVX2>
)
if(TERNAIR_HAS_AVX512)
    target_compile_options(ternair_native_cli PRIVATE -mavx512f -mavx512bw -mavx512dq)
elseif(TERNAIR_HAS_AVX2)
    target_compile_options(ternair_native_cli PRIVATE -mavx2 -mf16c -mfma)
endif()

# ---------------------------------------------------------------------------
# Tests (C++ smoke test)
# ---------------------------------------------------------------------------
enable_testing()
add_executable(ternair_native_test tests/test_matmul.cpp)
target_link_libraries(ternair_native_test PRIVATE ternair_native)
if(TERNAIR_HAS_AVX512)
    target_compile_options(ternair_native_test PRIVATE -mavx512f -mavx512bw -mavx512dq)
elseif(TERNAIR_HAS_AVX2)
    target_compile_options(ternair_native_test PRIVATE -mavx2 -mf16c -mfma)
endif()
add_test(NATIVE_MATMUL ternair_native_test)

install(TARGETS ternair_native ternair_native_cli
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
