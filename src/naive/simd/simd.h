#pragma once 

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void vec_add(float* A, const float* B, size_t N);
void vec_scalar_add(float* A, float b, size_t N);

void gemm(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K
);

void gemm_t1(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K
);

void gemm_t2(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K
);

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
);

// swap NxCxHxW => CxNxHxW
void swap_N_C(
    const float* X,
    float* Y,
    size_t N, size_t C, size_t H, size_t W
);

#ifdef __cplusplus
}
#endif
