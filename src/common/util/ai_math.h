#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// Adds a value to every element of v
void VectorAddScalar(float* v, float s, size_t size);

// Vector elementwise add: v1 += v2
void VectorAdd(float* v1, const float* v2, size_t size);

// dest = dest + scale * source
void VectorScaledAdd(float* dest, const float* source, float scale, size_t size);

// Vector elementwise sub: v1 -= v2
void VectorSub(float* v1, const float* v2, size_t size);

// Vector elementwise multiply
void VectorMul(float* v1, const float* v2, size_t size);

// Scale all elements of v by f: v *= f
void VectorScale(float* v, float f, size_t size);

// Copies all elements of v2 to v1
void VectorCopy(float* v1, const float* v2, size_t size);

// Sums all elements of v together
float Sum(const float* v, size_t size);

// Calculate the mean of a vector
float Mean(float* v, size_t size);

// Calculate the stddev of a vector (pass mean)
float StddevM(float* v, size_t size, float mean);

// Calculate the stddev of a vector (mean is calculated internally)
float Stddev(float* v, size_t size);
