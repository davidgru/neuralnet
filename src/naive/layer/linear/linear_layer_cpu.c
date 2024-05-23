#include "layer/linear/linear_layer_internal.h"

#if defined(USE_AVX)


#include <immintrin.h>


// AVX accelerated matrix product: output = m1 * m2
void matrix_product_cpu(const float* m1, const float* m2, float* output, size_t height1, size_t width2, size_t sharedDim)
{
    const size_t owidth = width2;
    const size_t oheight = height1;

    size_t c_unroll = owidth / 16 * 16;

    for (size_t r = 0; r < oheight; r++) {
        size_t c = 0;
        for (; c < c_unroll; c += 16) {
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            for (size_t s = 0; s < sharedDim; s++) {
                __m256 bc_m1 = _mm256_set1_ps(m1[r * sharedDim + s]);
                __m256 v1_m2 = _mm256_load_ps(m2 + s * width2 + c);
                __m256 v2_m2 = _mm256_load_ps(m2 + s * width2 + c + 8);
                __m256 prod1 = _mm256_mul_ps(bc_m1, v1_m2);
                __m256 prod2 = _mm256_mul_ps(bc_m1, v2_m2);
                sum1 = _mm256_add_ps(sum1, prod1);
                sum2 = _mm256_add_ps(sum2, prod2);
            }
            _mm256_storeu_ps(output + r * owidth + c, sum1);
            _mm256_storeu_ps(output + r * owidth + c + 8, sum2);
        }
        for (; c < owidth; c++) {
            float sum = 0.0f;
            for (size_t s = 0; s < sharedDim; s++)
                sum += m1[r * sharedDim + s] * m2[s * width2 + c];
            output[r * owidth + c] = sum;
        }
    }
}

// AVX accelerated matrix product, where m1 is transposed: output = m1_t * m2
void matrix_product_t1_cpu(const float* m1, const float* m2, float* output, size_t width1, size_t width2, size_t sharedDim)
{
    const size_t owidth = width2;
    const size_t oheight = width1;

    size_t c_unroll = owidth / 16 * 16;

    for (size_t r = 0; r < oheight; r++) {
        size_t c = 0;
        for (; c < c_unroll; c += 16) {
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            for (size_t s = 0; s < sharedDim; s++) {
                __m256 bc_m1 = _mm256_set1_ps(m1[s * width1 + r]);
                __m256 v1_m2 = _mm256_load_ps(m2 + s * width2 + c);
                __m256 v2_m2 = _mm256_load_ps(m2 + s * width2 + c + 8);
                __m256 prod1 = _mm256_mul_ps(bc_m1, v1_m2);
                __m256 prod2 = _mm256_mul_ps(bc_m1, v2_m2);
                sum1 = _mm256_add_ps(sum1, prod1);
                sum2 = _mm256_add_ps(sum2, prod2);
            }
            _mm256_storeu_ps(output + r * owidth + c, sum1);
            _mm256_storeu_ps(output + r * owidth + c + 8, sum2);
        }
        for (; c < owidth; c++) {
            float sum = 0.0f;
            for (size_t s = 0; s < sharedDim; s++)
                sum += m1[s * width1 + r] * m2[s * width2 + c];
            output[r * owidth + c] = sum;
        }
    }
}

// AVX accelerated matrix product, where m2 is transposed: output = m1 * m2_t
void matrix_product_t2_cpu(const float* m1, const float* m2, float* output, size_t height1, size_t height2, size_t sharedDim)
{
    const size_t owidth = height2;
    const size_t oheight = height1;

    size_t s_unroll = sharedDim / 8 * 8;

    for (size_t r = 0; r < oheight; r++) {
        for (size_t c = 0; c < owidth; c++) {
            __m256 vsum = _mm256_setzero_ps();
            size_t s = 0;
            for (; s < s_unroll; s += 8) {
                __m256 v1 = _mm256_loadu_ps(m1 + r * sharedDim + s);
                __m256 v2 = _mm256_loadu_ps(m2 + c * sharedDim + s);
                vsum = _mm256_add_ps(vsum, _mm256_mul_ps(v1, v2));
            }
            // Horizontal add
            __m256 t1 = _mm256_hadd_ps(vsum, vsum);
            __m256 t2 = _mm256_hadd_ps(t1, t1);
            __m128 t3 = _mm256_extractf128_ps(t2, 1);
            __m128 t4 = _mm_add_ss(_mm256_castps256_ps128(t2), t3);
            float sum = _mm_cvtss_f32(t4);

            for (; s < sharedDim; s++)
                sum += m1[r * sharedDim + s] * m2[c * sharedDim + s];
            output[r * owidth + c] = sum;
        }
    }
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
