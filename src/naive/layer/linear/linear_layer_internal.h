#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void matrix_product(const float* m1, const float* m2, float* output, size_t height1, size_t width2, size_t sharedDim);
void matrix_product_t1(const float* m1, const float* m2, float* output, size_t width1, size_t width2, size_t sharedDim);
void matrix_product_t2(const float* m1, const float* m2, float* output, size_t height1, size_t height2, size_t sharedDim);

#ifdef __cplusplus
}
#endif
