
extern "C" {
#include "simd.h"
}

#include "gemm/gemm.hpp"
#include "conv/im2col.hpp"
#include "conv/conv.hpp"
#include "common/vec_ops.hpp"

template<typename T, typename LayoutA, typename LayoutB>
static inline void gemm_impl(
    const T* A, const T* B, T* C,
    size_t M, size_t N, size_t K)
{
    // process in 128^3 blocks
    using BlockingPolicy = blocked<128, 128, 128>;

    // pack A & B
    using PackingPolicy = pack_ab;

    // 8x8 SIMD micorkernel
    using Kernel = kernel::avx2<T, 8, PackingPolicy::pack_B>;

    gemm<T, Kernel, BlockingPolicy, PackingPolicy,
        LayoutA, LayoutB>(A, B, C, M, N, K);
}

float vec_sum(const float* A,  size_t N)
{
    return vec_sum<float, 64>(A, N);
}

void vec_add(float* A, const float* B, size_t N)
{
    vec_add<float>(A, B, N);
}

void vec_scalar_add(float* A, float b, size_t N)
{
    vec_scalar_add<float>(A, b, N);
}

void gemm(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K
) {
    // data type is float
    using T = float;

    // A & B assumed row-major
    using LayoutA = row_major_t;
    using LayoutB = row_major_t;

    gemm_impl<T, LayoutA, LayoutB>(A, B, C, M, N, K);
}

void gemm_t1(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K
) {
    // data type is float
    using T = float;

    // A transposed (col-major) & B assumed row-major
    using LayoutA = col_major_t;
    using LayoutB = row_major_t;

    gemm_impl<T, LayoutA, LayoutB>(A, B, C, M, N, K);
}

void gemm_t2(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K
) {
    // data type is float
    using T = float;

    // A assumed row-major & B transposed (col-major)
    using LayoutA = row_major_t;
    using LayoutB = col_major_t;

    gemm_impl<T, LayoutA, LayoutB>(A, B, C, M, N, K);
}

void conv(
    const float* X, // NxICxIHxIW input
    const float* F, // OCxICxKHxKW filter
    float* Y, // NxOCxOHxOW output
    size_t N, size_t IC, size_t IH, size_t IW,
    size_t OC, size_t KH, size_t KW,
    size_t PH, size_t PW,
    size_t SH, size_t SW,
    size_t DH, size_t DW,
    size_t CROP_H, size_t CROP_W
) {
    // data type is float
    using T = float;

    const size_t OH = cod(IH, KH, PH, SH, DH, CROP_H);
    const size_t OW = cod(IW, KW, PW, SW, DW, CROP_W);

    // For big kernels using transposed Im2Col is more cache-favorable
    if (KH*KW > (OH/2)*(OW/2)) {
        // GEMM implicitly transposes B
        using LayoutA = row_major_t;
        using LayoutB = col_major_t;

        // Im2Col_T: X: NxICxIHxIW => M: (N*OH*OW)x(IC*KH*KW)
        // GEMM_T2: F * M^T => OCx(N*OH*OW)
        // Reshape: OCx(N*OH*OW) => NxOCxOHxOC
        conv<T, im2col_T<T>,
             gemm_impl<T, LayoutA, LayoutB>>(
           X, F, Y, N, IC, IH, IW, OC, KH, KW,
           PH, PW, SH, SW, DH, DW, CROP_H, CROP_W
        );
    } else {
        // Normal GEMM
        using LayoutA = row_major_t;
        using LayoutB = row_major_t;
        
        // Im2Col: X: NxICxIHxIW => M: (IC*KH*KW)x(N*OH*OW)
        // GEMM: F * M => Y: OCx(N*OH*OW)
        // Reshape: OCx(N*OH*OW) => NxOCxOHxOC
        conv<T, im2col<T>,
             gemm_impl<T, LayoutA, LayoutB>>(
           X, F, Y, N, IC, IH, IW, OC, KH, KW,
           PH, PW, SH, SW, DH, DW, CROP_H, CROP_W
        );
    }
}

// swap NxCxHxW => CxNxHxW
void swap_N_C(
    const float* X,
    float* Y,
    size_t N, size_t C, size_t H, size_t W
) {
    swap_N_C<float>(X, Y, N, C, H, W);
}
