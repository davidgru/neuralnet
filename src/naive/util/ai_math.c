
#include "ai_math.h"

#include <math.h>
#include <string.h>
#if defined(AI_USE_AVX)
#include <immintrin.h>
#endif


#define MIN(a, b) (((a) < (b)) ? a : b)


#if defined(AI_USE_AVX)


void AI_VectorAdd(float* v1, float* v2, size_t size)
{
    size_t s_unroll = size / 8 * 8;
    size_t i = 0;
    for (; i < s_unroll; i += 8) {
        __m256 _v1 = _mm256_loadu_ps(v1 + i);
        __m256 _v2 = _mm256_loadu_ps(v2 + i);
        _v1 = _mm256_add_ps(_v1, _v2);
        _mm256_storeu_ps(v1 + i, _v1);
    }
    for (; i < size; i++)
        v1[i] += v2[i];
}


void AI_VectorSub(float* v1, float* v2, size_t size)
{
    size_t s_unroll = size / 8 * 8;
    size_t i = 0;
    for (; i < s_unroll; i += 8) {
        __m256 _v1 = _mm256_loadu_ps(v1 + i);
        __m256 _v2 = _mm256_loadu_ps(v2 + i);
        _v1 = _mm256_sub_ps(_v1, _v2);
        _mm256_storeu_ps(v1 + i, _v1);
    }
    for (; i < size; i++)
        v1[i] -= v2[i];
}


void AI_VectorMul(float* v1, float* v2, size_t size)
{
    size_t s_unroll = size / 8 * 8;
    size_t i = 0;
    for (; i < s_unroll; i += 8) {
        __m256 _v1 = _mm256_loadu_ps(v1 + i);
        __m256 _v2 = _mm256_loadu_ps(v2 + i);
        _v1 = _mm256_mul_ps(_v1, _v2);
        _mm256_storeu_ps(v1 + i, _v1);
    }
    for (; i < size; i++)
        v1[i] *= v2[i];
}


void AI_VectorScale(float* v, float f, size_t size)
{
    size_t s_unroll = size / 8 * 8;
    size_t i = 0;
    __m256 scalar = _mm256_set1_ps(f);
    for (; i < s_unroll; i += 8) {
        __m256 _v = _mm256_loadu_ps(v + i);
        _v = _mm256_mul_ps(_v, scalar);
        _mm256_storeu_ps(v + i, _v);
    }
}


#else


void AI_VectorAdd(float* v1, float* v2, size_t size)
{
    for (size_t i = 0; i < size; i++)
        v1[i] += v2[i];
}


void AI_VectorSub(float* v1, float* v2, size_t size)
{
    for (size_t i = 0; i < size; i++)
        v1[i] -= v2[i];
}


void AI_VectorMul(float* v1, float* v2, size_t size)
{
    for (size_t i = 0; i < size; i++)
        v1[i] *= v2[i];    
}


void AI_VectorScale(float* v, float f, size_t size)
{
    for (size_t i = 0; i < size; i++)
        v[i] *= f;
}


#endif


void AI_VectorAddScalar(float* v, float s, size_t size)
{
    for (size_t i = 0; i < size; i++)
        v[i] += s;
}


// Copies all elements of v2 to v1
void AI_VectorCopy(float* v1, float* v2, size_t size)
{
    for (size_t i = 0; i < size; i++)
        v1[i] = v2[i];
}


// Convolution operation
void AI_MatrixConvolution(float* m1, float* kernel, float* output, size_t mwidth, size_t mheight, size_t kwidth, size_t kheight, size_t strideX, size_t strideY)
{
    size_t owidth = (mwidth - kwidth) / strideX + 1;
    size_t oheight = (mheight - kheight) / strideY + 1;

    for (size_t r = 0; r < oheight; r++)
        for (size_t kr = 0; kr < kheight; kr++)
            for (size_t kc = 0; kc < kwidth; kc++)
                for (size_t c = 0; c < owidth; c++)
                    output[r * owidth + c] += m1[(r * strideY + kr) * mwidth + (c * strideX + kc)] * kernel[kr * kwidth + kc];
}


// Convolution operation with padding and dilation
void AI_MatrixConvolutionPadded(float* m1, float* kernel, float* output, size_t mwidth, size_t mheight, size_t kwidth, size_t kheight, size_t strideX, size_t strideY, size_t padX, size_t padY, size_t dilationX, size_t dilationY)
{
    size_t owidth = (mwidth - kwidth + 2 * padX) / strideX + 1;
    size_t oheight = (mheight - kheight + 2 * padY) / strideY + 1;

    for (size_t r = 0; r < oheight; r++) {
        for (size_t c = 0; c < owidth; c++) {
            float sum = 0.0f;
            for (size_t kr = 0; kr < kheight; kr++) {
                for (size_t kc = 0; kc < kwidth; kc++) {
                    int32_t dataR = (int32_t)r * strideY + kr;
                    int32_t dataC = (int32_t)c * strideX + kc;
                    dataR = dataR - padY;
                    dataC = dataC - padX;
                    if (dataR > -1 && dataC > -1 && dataR < mheight && dataC < mwidth)
                        sum += m1[dataR * mwidth + dataC] * kernel[kr * kwidth + kc];
                }
            }
            output[r * owidth + c] += sum;
        }
    }
}


// Convolution operation with padding and dilation and a 180° rotated kernel
void AI_MatrixConvolutionPaddedRotateFilter(float* m1, float* kernel, float* output, size_t mwidth, size_t mheight, size_t kwidth, size_t kheight, size_t strideX, size_t strideY, size_t padX, size_t padY, size_t dilationX, size_t dilationY)
{
    size_t owidth = (mwidth - kwidth + 2 * padX) / strideX + 1;
    size_t oheight = (mheight - kheight + 2 * padY) / strideY + 1;

    for (size_t r = 0; r < oheight; r++) {
        for (size_t kr = 0; kr < kheight; kr++) {
            for (size_t kc = 0; kc < kwidth; kc++) {
                for (size_t c = 0; c < owidth; c++) {
                    int32_t dataR = (int32_t)r * strideY + kr;
                    int32_t dataC = (int32_t)c * strideX + kc;
                    dataR = dataR - padY;
                    dataC = dataC - padX;
                    if (dataR > -1 && dataC > -1 && dataR < mheight && dataC < mwidth)
                            output[r * owidth + c] += m1[dataR * mwidth + dataC] * kernel[(kheight - kr - 1) * kwidth + (kwidth - kc - 1)];
                }
            }
        }
    }
}


// Sums all elements of v together
float AI_Sum(float* v, size_t size)
{
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++)
        sum += v[i];
    return sum;
}


// Calculate the mean of a vector
float AI_Mean(float* v, size_t size)
{
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        sum += v[i];
    }
    return sum / size;
}


// Calculate the stddev of a vector (pass mean)
float AI_StddevM(float* v, size_t size, float mean)
{
    float squared_sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        squared_sum += (v[i] - mean) * (v[i] - mean);
    }
    squared_sum /= (float)size;
    return sqrtf(squared_sum);
}


// Calculate the stddev of a vector (mean is calculated internally)
float AI_Stddev(float* v, size_t size)
{
    float mean = AI_Mean(v, size);
    return AI_StddevM(v, size, mean);
}
