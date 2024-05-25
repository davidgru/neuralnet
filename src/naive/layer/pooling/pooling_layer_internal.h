#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "layer/pooling_layer.h"
#include "tensor/tensor_impl.h"

#define tensor_dim(tensor, dim)         ((tensor)->shape.dims[dim])
#define tensor_batch_size(tensor)       (tensor_dim(tensor, TENSOR_BATCH_DIM))
#define tensor_channels(tensor)         (tensor_dim(tensor, TENSOR_CHANNEL_DIM))
#define tensor_height(tensor)           (tensor_dim(tensor, TENSOR_HEIGHT_DIM))
#define tensor_width(tensor)            (tensor_dim(tensor, TENSOR_WIDTH_DIM))
#define tensor_per_channel_size(tensor) (tensor_height(tensor) * tensor_width(tensor))
#define tensor_per_batch_size(tensor)   (tensor_channels(tensor) * tensor_height(tensor) * tensor_width(tensor))


void pooling_forward_cpu(const tensor_t* input, tensor_t* output, size_t kernel_width, pooling_kind_t kind);
void pooling_backward_cpu(const tensor_t* input, const tensor_t* prev_grad, tensor_t* grad, size_t kernel_width, pooling_kind_t kind);

#if defined(USE_GPU)
void pooling_forward_gpu(const tensor_t* input, tensor_t* output, size_t kernel_width, pooling_kind_t kind);
void pooling_backward_gpu(const tensor_t* input, const tensor_t* prev_grad, tensor_t* grad, size_t kernel_width, pooling_kind_t kind);
#endif

#ifdef __cplusplus
}
#endif
