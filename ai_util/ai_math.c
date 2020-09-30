
#include "ai_math.h"

#include <math.h>
#include <string.h>
#include <immintrin.h>

#define MIN(a, b) (((a) < (b)) ? a : b)

// Adds a value to every element of v
void AI_VectorAddScalar(float* v, float s, size_t size)
{
    for (size_t i = 0; i < size; i++)
        v[i] += s;
}

// Vector elementwise add: v1 += v2
void AI_VectorAdd(float* v1, float* v2, size_t size)
{
    for (size_t i = 0; i < size; i++)
        v1[i] += v2[i];
}

void AI_VectorAddAVX(float* v1, float* v2, size_t size)
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

// Vector elementwise sub: v1 -= v2
void AI_VectorSub(float* v1, float* v2, size_t size)
{
    for (size_t i = 0; i < size; i++)
        v1[i] -= v2[i];
}

void AI_VectorSubAVX(float* v1, float* v2, size_t size)
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


// Vector elementwise multiply
void AI_VectorMul(float* v1, float* v2, size_t size)
{
    for (size_t i = 0; i < size; i++)
        v1[i] *= v2[i];    
}

void AI_VectorMulAVX(float* v1, float* v2, size_t size)
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


// Scale all elements of v by f: v *= f
void AI_VectorScale(float* v, float f, size_t size)
{
    for (size_t i = 0; i < size; i++)
        v[i] *= f;
}

void AI_VectorScaleAVX(float* v, float f, size_t size)
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


// Copies all elements of v2 to v1
void AI_VectorCopy(float* v1, float* v2, size_t size)
{
    for (size_t i = 0; i < size; i++)
        v1[i] = v2[i];
}


// Matrix multiplication: output = m1 * m2
void AI_MatrixDotProduct(float* m1, float* m2, float* output, size_t height1, size_t width2, size_t sharedDim)
{
    const size_t owidth = width2;
    const size_t oheight = height1;

    for (size_t r = 0; r < oheight; r++) {
        for (size_t c = 0; c < owidth; c++) {
            float sum = 0.0f;
            for (size_t s = 0; s < sharedDim; s++)
                sum += m1[r * sharedDim + s] * m2[s * width2 + c];
            output[r * owidth + c] = sum;
        }
    }
}

// Matrix multiplication where t1 is transposed: output = m1_T * m2
void AI_MatrixDotProductT1(float* m1, float* m2, float* output, size_t width1, size_t width2, size_t sharedDim)
{
    const size_t owidth = width2;
    const size_t oheight = width1;

    for (size_t r = 0; r < oheight; r++) {
        for (size_t c = 0; c < owidth; c++) {
            float sum = 0.0f;
            for (size_t s = 0; s < sharedDim; s++)
                sum += m1[s * width1 + r] * m2[s * width2 + c];
            output[r * owidth + c] = sum;
        }
    }
}

// Matrix multiplication where t2 is transposed: output = m1 * m2_T
void AI_MatrixDotProductT2(float* m1, float* m2, float* output, size_t height1, size_t height2, size_t sharedDim)
{
    const size_t owidth = height2;
    const size_t oheight = height1;

    for (size_t r = 0; r < oheight; r++) {
        for (size_t c = 0; c < owidth; c++) {
            float sum = 0.0f;
            for (size_t s = 0; s < sharedDim; s++)
                sum += m1[r * sharedDim + s] * m2[c * sharedDim + s];
            output[r * owidth + c] = sum;
        }
    }
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
    for (size_t i = 0; i < size; i++)
        squared_sum += v[i] * v[i];
    return (squared_sum / size) - (mean * mean);
}

// Calculate the stddev of a vector (mean is calculated internally)
float AI_Stddev(float* v, size_t size)
{
    float sum = 0.0f;
    float squared_sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        sum += v[i];
        squared_sum += v[i] * v[i];
    }
    return sqrtf((squared_sum / size) - (sum * sum) / (size * size));
}
