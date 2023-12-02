
#include <math.h>
#include <string.h>
#if defined(USE_AVX)
#include <immintrin.h>
#endif

#include "ai_math.h"


#define MIN(a, b) (((a) < (b)) ? a : b)


#if defined(USE_AVX)


void VectorAdd(float* v1, const float* v2, size_t size)
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


void VectorScaledAdd(float* v1, const float* v2, float scale, size_t size)
{
    size_t s_unroll = size / 8 * 8;
    size_t i = 0;

    __m256 _scale = _mm256_set1_ps(scale);

    for (; i < s_unroll; i += 8) {
        __m256 _v1 = _mm256_loadu_ps(v1 + i);
        __m256 _v2 = _mm256_loadu_ps(v2 + i);
        _v2 = _mm256_mul_ps(_v2, _scale);
        _v1 = _mm256_add_ps(_v1, _v2);
        _mm256_storeu_ps(v1 + i, _v1);
    }
    for (; i < size; i++)
        v1[i] += scale * v2[i];
}


void VectorSub(float* v1, const float* v2, size_t size)
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


void VectorMul(float* v1, const float* v2, size_t size)
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


void VectorScale(float* v, float f, size_t size)
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


void VectorAdd(float* v1, const float* v2, size_t size)
{
    for (size_t i = 0; i < size; i++)
        v1[i] += v2[i];
}

void VectorScaledAdd(float* dest, const float* source, float scale, size_t size)
{
    for (size_t i = 0; i < size; i++)
        dest[i] += scale * source[i];
}


void VectorSub(float* v1, const float* v2, size_t size)
{
    for (size_t i = 0; i < size; i++)
        v1[i] -= v2[i];
}


void VectorMul(float* v1, const float* v2, size_t size)
{
    for (size_t i = 0; i < size; i++)
        v1[i] *= v2[i];    
}


void VectorScale(float* v, float f, size_t size)
{
    for (size_t i = 0; i < size; i++)
        v[i] *= f;
}


#endif


void VectorAddScalar(float* v, float s, size_t size)
{
    for (size_t i = 0; i < size; i++)
        v[i] += s;
}


// Copies all elements of v2 to v1
void VectorCopy(float* v1, const float* v2, size_t size)
{
    for (size_t i = 0; i < size; i++)
        v1[i] = v2[i];
}



// Sums all elements of v together
float Sum(const float* v, size_t size)
{
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++)
        sum += v[i];
    return sum;
}


// Calculate the mean of a vector
float Mean(float* v, size_t size)
{
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        sum += v[i];
    }
    return sum / size;
}


// Calculate the stddev of a vector (pass mean)
float StddevM(float* v, size_t size, float mean)
{
    float squared_sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        squared_sum += (v[i] - mean) * (v[i] - mean);
    }
    squared_sum /= (float)size;
    return sqrtf(squared_sum);
}


// Calculate the stddev of a vector (mean is calculated internally)
float Stddev(float* v, size_t size)
{
    float mean = Mean(v, size);
    return StddevM(v, size, mean);
}
