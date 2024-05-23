#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void matrix_product_cpu(const float* m1, const float* m2, float* output, size_t height1, size_t width2, size_t sharedDim);
void matrix_product_t1_cpu(const float* m1, const float* m2, float* output, size_t width1, size_t width2, size_t sharedDim);
void matrix_product_t2_cpu(const float* m1, const float* m2, float* output, size_t height1, size_t height2, size_t sharedDim);

#if defined(USE_GPU)
void matrix_product_gpu(const float* m1, const float* m2, float* output, size_t height1, size_t width2, size_t sharedDim);
void matrix_product_t1_gpu(const float* m1, const float* m2, float* output, size_t width1, size_t width2, size_t sharedDim);
void matrix_product_t2_gpu(const float* m1, const float* m2, float* output, size_t height1, size_t height2, size_t sharedDim);
#endif

#ifdef __cplusplus
}
#endif
