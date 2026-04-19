#pragma once 

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif
