#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "tensor/tensor_impl.h"


#define tensor_dim(tensor, dim)         ((tensor)->shape.dims[dim])
#define tensor_batch_size(tensor)       (tensor_dim(tensor, TENSOR_BATCH_DIM))
#define tensor_channels(tensor)         (tensor_dim(tensor, TENSOR_CHANNEL_DIM))
#define tensor_height(tensor)           (tensor_dim(tensor, TENSOR_HEIGHT_DIM))
#define tensor_width(tensor)            (tensor_dim(tensor, TENSOR_WIDTH_DIM))
#define tensor_per_channel_size(tensor) (tensor_height(tensor) * tensor_width(tensor))
#define tensor_per_batch_size(tensor)   (tensor_channels(tensor) * tensor_height(tensor) * tensor_width(tensor))
#define _filter_height(filter)           (tensor_dim(filter, CONV_WEIGHT_HEIGHT_DIM))
#define _filter_width(filter)            (tensor_dim(filter, CONV_WEIGHT_WIDTH_DIM))
#define _filter_size(filter)             (tensor_dim(filter, CONV_WEIGHT_INPUT_CHANNEL_DIM) * _filter_height(filter) * _filter_width(filter))

void batchnorm_forward_cpu(const tensor_t* input, const tensor_t* mean, const tensor_t* var,
    const tensor_t* scale, const tensor_t* shift, tensor_t* output, float bn_eps);
void batchnorm_backward_weights_cpu(const tensor_t* input, const tensor_t* mean, const tensor_t* var,
    const tensor_t* d_output, tensor_t* d_scale, tensor_t* d_shift, float bn_eps);
void batchnorm_backward_data_cpu(const tensor_t* input, const tensor_t* mean, const tensor_t* var,
    const tensor_t* scale, const tensor_t* d_output, tensor_t* tmp_d_mean, tensor_t* tmp_d_var,
    tensor_t* d_input, float bn_eps);

#if defined(USE_GPU)

void batchnorm_forward_gpu(const tensor_t* input, const tensor_t* mean, const tensor_t* var,
    const tensor_t* scale, const tensor_t* shift, tensor_t* output, float bn_eps);
void batchnorm_backward_weights_gpu(const tensor_t* input, const tensor_t* mean, const tensor_t* var,
    const tensor_t* d_output, tensor_t* d_scale, tensor_t* d_shift, float bn_eps, tensor_t* tmp1, tensor_t* tmp2);
void batchnorm_backward_data_gpu(const tensor_t* input, const tensor_t* mean, const tensor_t* var,
    const tensor_t* scale, const tensor_t* d_output, tensor_t* tmp_d_mean, tensor_t* tmp_d_var,
    tensor_t* d_input, float bn_eps, tensor_t* tmp1, tensor_t* tmp2);

#endif

#ifdef __cplusplus
}
#endif
