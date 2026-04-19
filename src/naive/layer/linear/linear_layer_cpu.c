#include "layer/linear/linear_layer_internal.h"

#if defined(USE_AVX)

#include <memory.h>

#include "simd/simd.h"

// AVX accelerated matrix product: output = m1 * m2
void matrix_product_cpu(const float* m1, const float* m2, float* output, size_t height1, size_t width2, size_t sharedDim)
{
    /* GEMM assumes zeroed memory. */
    memset(output, 0, height1 * width2 * sizeof(float));

    gemm(m1, m2, output, height1, width2, sharedDim);
}

// AVX accelerated matrix product, where m1 is transposed: output = m1_t * m2
void matrix_product_t1_cpu(const float* m1, const float* m2, float* output, size_t width1, size_t width2, size_t sharedDim)
{
    /* GEMM assumes zeroed memory. */
    memset(output, 0, width1 * width2 * sizeof(float));

    gemm_t1(m1, m2, output, width1, width2, sharedDim);
}

// AVX accelerated matrix product, where m2 is transposed: output = m1 * m2_t
void matrix_product_t2_cpu(const float* m1, const float* m2, float* output, size_t height1, size_t height2, size_t sharedDim)
{
    /* GEMM assumes zeroed memory. */
    memset(output, 0, height1 * height2 * sizeof(float));

    gemm_t2(m1, m2, output, height1, height2, sharedDim);
}


#else


void matrix_product_cpu(const float* m1, const float* m2, float* output, size_t height1, size_t width2, size_t sharedDim)
{
    const size_t owidth = width2;
    const size_t oheight = height1;

    for (size_t r = 0; r < oheight; r++) {
        for (size_t c = 0; c < owidth; c++) {
            float sum = 0.0f;
            for (size_t s = 0; s < sharedDim; s++) {
                sum += m1[r * sharedDim + s] * m2[s * width2 + c];
            }
            output[r * owidth + c] = sum;
        }
    }
}

void matrix_product_t1_cpu(const float* m1, const float* m2, float* output, size_t width1, size_t width2, size_t sharedDim)
{
    const size_t owidth = width2;
    const size_t oheight = width1;

    for (size_t r = 0; r < oheight; r++) {
        for (size_t c = 0; c < owidth; c++) {
            float sum = 0.0f;
            for (size_t s = 0; s < sharedDim; s++) {
                sum += m1[s * width1 + r] * m2[s * width2 + c];
            }
            output[r * owidth + c] = sum;
        }
    }
}

void matrix_product_t2_cpu(const float* m1, const float* m2, float* output, size_t height1, size_t height2, size_t sharedDim)
{
    const size_t owidth = height2;
    const size_t oheight = height1;

    for (size_t r = 0; r < oheight; r++) {
        for (size_t c = 0; c < owidth; c++) {
            float sum = 0.0f;
            for (size_t s = 0; s < sharedDim; s++) {
                sum += m1[r * sharedDim + s] * m2[c * sharedDim + s];
            }
            output[r * owidth + c] = sum;
        }
    }
}


#endif
