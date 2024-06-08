#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "tensor/tensor.h"
#include "tensor/tensor_math.h"


void tensor_scale_cpu(tensor_t* v, float f);
void tensor_add_scalar_cpu(tensor_t* v, float f);
void tensor_eltwise_add_cpu(tensor_t* v, const tensor_t* w);
void tensor_eltwise_mul_cpu(tensor_t* v, const tensor_t* w);
void tensor_scaled_add_cpu(tensor_t* v, const tensor_t* w, float f);
void tensor_sum_cpu(tensor_t* v, const tensor_t* w);
void tensor_sum_axis_cpu(tensor_t* v, const tensor_t* w, size_t outer_stride,
    size_t outer_len, size_t axis_len, size_t inner_stride);
void tensor_variance_axis_cpu(tensor_t* v, const tensor_t* w, const tensor_t* mean,
    size_t outer_stride, size_t outer_len, size_t axis_len, size_t inner_stride);
void tensor_random_mask_cpu(tensor_t* v, float ratio);
void tensor_momentum_update_cpu(tensor_t* v, const tensor_t* w, float momentum);

#if defined(USE_GPU)
void tensor_scale_gpu(tensor_t* v, float f);
void tensor_add_scalar_gpu(tensor_t* v, float f);
void tensor_eltwise_add_gpu(tensor_t* v, const tensor_t* w);
void tensor_eltwise_mul_gpu(tensor_t* v, const tensor_t* w);
void tensor_scaled_add_gpu(tensor_t* v, const tensor_t* w, float f);
void tensor_sum_gpu(tensor_t* v, const tensor_t* w);
void tensor_sum_axis_gpu(tensor_t* v, const tensor_t* w, size_t outer_stride,
    size_t outer_len, size_t axis_len, size_t inner_stride);
void tensor_variance_axis_gpu(tensor_t* v, const tensor_t* w, const tensor_t* mean,
    size_t outer_stride, size_t outer_len, size_t axis_len, size_t inner_stride);
void tensor_random_mask_gpu(tensor_t* v, float ratio);
void tensor_momentum_update_gpu(tensor_t* v, const tensor_t* w, float momentum);
#endif

#ifdef __cplusplus
}
#endif
