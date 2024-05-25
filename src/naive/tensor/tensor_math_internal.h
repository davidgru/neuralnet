#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "tensor/tensor.h"

void tensor_scale_cpu(tensor_t* v, float f);
void tensor_add_scalar_cpu(tensor_t* v, float f);
void tensor_eltwise_add_cpu(tensor_t* v, const tensor_t* w);
void tensor_eltwise_mul_cpu(tensor_t* v, const tensor_t* w);
void tensor_scaled_add_cpu(tensor_t* v, const tensor_t* w, float f);
void tensor_sum_cpu(tensor_t* v, const tensor_t* w);

#if defined(USE_GPU)
void tensor_scale_gpu(tensor_t* v, float f);
void tensor_add_scalar_gpu(tensor_t* v, float f);
void tensor_eltwise_add_gpu(tensor_t* v, const tensor_t* w);
void tensor_eltwise_mul_gpu(tensor_t* v, const tensor_t* w);
void tensor_scaled_add_gpu(tensor_t* v, const tensor_t* w, float f);
void tensor_sum_gpu(tensor_t* v, const tensor_t* w);
#endif

#ifdef __cplusplus
}
#endif