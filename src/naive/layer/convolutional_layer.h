#pragma once

#include "core/layer.h"

#include "util/weight_init.h"


typedef struct {
    size_t output_channels;
    size_t filter_height;
    size_t filter_width;
    size_t stride_y;
    size_t stride_x;
    size_t padding_y;
    size_t padding_x;
    conv_weight_init_func_t weight_init;
    conv_bias_init_func_t bias_init;
} convolutional_layer_create_info_t;


extern const layer_impl_t convolutional_layer_impl;
