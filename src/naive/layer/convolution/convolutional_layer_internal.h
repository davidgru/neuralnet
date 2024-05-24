#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "tensor/tensor_impl.h"


#define CONV_WEIGHT_OUTPUT_CHANNEL_DIM  0
#define CONV_WEIGHT_INPUT_CHANNEL_DIM   1
#define CONV_WEIGHT_HEIGHT_DIM          2
#define CONV_WEIGHT_WIDTH_DIM           3

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

#define conv_output_size(input_size, kernel_size, stride, dilation, padding) \
    (((input_size) + 2 * (padding) - (dilation) * ((kernel_size) - 1) - 1) / (stride) + 1)
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define div_ceil(a, b) (((a) + (b) - 1) / (b))

void conv2d_cpu(const float* input, const float* kernel, float* output, int32_t input_height,
    int32_t input_width, int32_t kernel_height, int32_t kernel_width, int32_t stride_y,
    int32_t stride_x, int32_t padding_y, int32_t padding_x, int32_t dilation_y, int32_t dilation_x,
    int32_t skip_output_y, int32_t skip_output_x, int32_t flip_kernel);

void convolution_forward_cpu(const tensor_t* input, const tensor_t* filter, const tensor_t* bias,
    tensor_t* output, int32_t stride_y, int32_t stride_x, int32_t padding_y, int32_t padding_x,
    int32_t dilation_y, int32_t dilation_x, int32_t skip_output_y, int32_t skip_output_x,
    int32_t flip_kernel);

void convolution_backward_data_cpu(const tensor_t* prev_grad, const tensor_t* filter, tensor_t* grad,
    int32_t stride_y, int32_t stride_x, int32_t padding_y, int32_t padding_x);

#if defined(USE_GPU)

void conv2d_gpu(const float* input, const float* kernel, float* output, int32_t input_height,
    int32_t input_width, int32_t kernel_height, int32_t kernel_width, int32_t stride_y,
    int32_t stride_x, int32_t padding_y, int32_t padding_x, int32_t dilation_y, int32_t dilation_x,
    int32_t skip_output_y, int32_t skip_output_x, int32_t flip_kernel);

void convolution_forward_gpu(const tensor_t* input, const tensor_t* filter, const tensor_t* bias,
    tensor_t* output, int32_t stride_y, int32_t stride_x, int32_t padding_y, int32_t padding_x,
    int32_t dilation_y, int32_t dilation_x, int32_t skip_output_y, int32_t skip_output_x,
    int32_t flip_kernel);

void convolution_backward_data_gpu(const tensor_t* prev_grad, const tensor_t* filter, tensor_t* grad,
    int32_t stride_y, int32_t stride_x, int32_t padding_y, int32_t padding_x);

#endif

#ifdef __cplusplus
}
#endif
