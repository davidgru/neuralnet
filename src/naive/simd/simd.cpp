
extern "C" {
#include "simd.h"
}

#include "gemm/gemm.hpp"

void gemm(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K
) {
    // data type is float
    using T = float;

    // process in 128^3 blocks
    using BlockingPolicy = blocked<128, 128, 128>;

    // pack A & B
    using PackingPolicy = pack_ab;

    // A & B assumed row-major
    using LayoutA = row_major_t;
    using LayoutB = row_major_t;

    // 8x8 SIMD micorkernel
    using Kernel = kernel::avx2<T, 8, PackingPolicy::pack_B>;

    gemm<T, Kernel, BlockingPolicy, PackingPolicy,
        LayoutA, LayoutB>(A, B, C, M, N, K);
}

void gemm_t1(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K
) {
    // data type is float
    using T = float;

    // process in 128^3 blocks
    using BlockingPolicy = blocked<128, 128, 128>;

    // pack A & B
    using PackingPolicy = pack_ab;

    // A transposed (col-major) & B assumed row-major
    using LayoutA = col_major_t;
    using LayoutB = row_major_t;

    // 8x8 SIMD micorkernel
    using Kernel = kernel::avx2<T, 8, PackingPolicy::pack_B>;

    gemm<T, Kernel, BlockingPolicy, PackingPolicy,
        LayoutA, LayoutB>(A, B, C, M, N, K);

}

void gemm_t2(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K
) {
    // data type is float
    using T = float;

    // process in 128^3 blocks
    using BlockingPolicy = blocked<128, 128, 128>;

    // pack A & B
    using PackingPolicy = pack_ab;

    // A assumed row-major & B transposed (col-major)
    using LayoutA = row_major_t;
    using LayoutB = col_major_t;

    // 8x8 SIMD micorkernel
    using Kernel = kernel::avx2<T, 8, PackingPolicy::pack_B>;

    gemm<T, Kernel, BlockingPolicy, PackingPolicy,
        LayoutA, LayoutB>(A, B, C, M, N, K);
}
