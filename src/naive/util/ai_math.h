#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// Adds a value to every element of v
void AI_VectorAddScalar(float* v, float s, size_t size);

// Vector elementwise add: v1 += v2
void AI_VectorAdd(float* v1, float* v2, size_t size);
void AI_VectorAddAVX(float* v1, float* v2, size_t size);

// Vector elementwise sub: v1 -= v2
void AI_VectorSub(float* v1, float* v2, size_t size);
void AI_VectorSubAVX(float* v1, float* v2, size_t size);

// Vector elementwise multiply
void AI_VectorMul(float* v1, float* v2, size_t size);
void AI_VectorMulAVX(float* v1, float* v2, size_t size);

// Scale all elements of v by f: v *= f
void AI_VectorScale(float* v, float f, size_t size);
void AI_VectorScaleAVX(float* v, float f, size_t size);

// Copies all elements of v2 to v1
void AI_VectorCopy(float* v1, float* v2, size_t size);


// Matrix multiplication: output = m1 * m2
void AI_MatrixDotProduct(float* m1, float* m2, float* output, size_t height1, size_t width2, size_t sharedDim);

// Matrix multiplication where t1 is transposed: output = m1_T * m2
void AI_MatrixDotProductT1(float* m1, float* m2, float* output, size_t width1, size_t width2, size_t sharedDim);

// Matrix multiplication where t2 is transposed: output = m1 * m2_T
void AI_MatrixDotProductT2(float* m1, float* m2, float* output, size_t height1, size_t height2, size_t sharedDim);


// Convolution operation
void AI_MatrixConvolution(float* m1, float* kernel, float* output, size_t mwidth, size_t mheight, size_t kwidth, size_t kheight, size_t strideX, size_t strideY);

// Convolution operation with padding and dilation
void AI_MatrixConvolutionPadded(float* m1, float* kernel, float* output, size_t mwidth, size_t mheight, size_t kwidth, size_t kheight, size_t strideX, size_t strideY, size_t padX, size_t padY, size_t dilationX, size_t dilationY);


// Convolution operation with padding and dilation and a by 180° rotated kernel
void AI_MatrixConvolutionPaddedRotateFilter(float* m1, float* kernel, float* output, size_t mwidth, size_t mheight, size_t kwidth, size_t kheight, size_t strideX, size_t strideY, size_t padX, size_t padY, size_t dilationX, size_t dilationY);


// Sums all elements of v together
float AI_Sum(float* v, size_t size);

// Calculate the mean of a vector
float AI_Mean(float* v, size_t size);

// Calculate the stddev of a vector (pass mean)
float AI_StddevM(float* v, size_t size, float mean);

// Calculate the stddev of a vector (mean is calculated internally)
float AI_Stddev(float* v, size_t size);
