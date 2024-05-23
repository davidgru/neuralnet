#pragma once

#include "tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

void sigmoid_cpu(const float* in, float* out, size_t size);
void tanh_cpu(const float* in, float* out, size_t size);
void relu_cpu(const float* in, float* out, size_t size);

void dsigmoid_cpu(const float* in, float* out, size_t size);
void dtanh_cpu(const float* in, float* out, size_t size);
void drelu_cpu(const float* in, float* out, size_t size);

#if defined(USE_GPU)
void sigmoid_gpu(const float* in, float* out, size_t size);
void tanh_gpu(const float* in, float* out, size_t size);
void relu_gpu(const float* in, float* out, size_t size);

void dsigmoid_gpu(const float* in, float* out, size_t size);
void dtanh_gpu(const float* in, float* out, size_t size);
void drelu_gpu(const float* in, float* out, size_t size);
#endif

#ifdef __cplusplus
}
#endif
