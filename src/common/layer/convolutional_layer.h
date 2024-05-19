#pragma once

#include "core/layer.h"
#include "core/layer_impl.h"

#include "util/weight_init.h"


typedef struct {
    size_t output_channels;
    size_t filter_height;
    size_t filter_width;
    size_t stride_y;
    size_t stride_x;
    size_t padding_y;
    size_t padding_x;
    weight_init_func_t weight_init;
    weight_init_func_t bias_init;
} convolutional_layer_create_info_t;


extern const convolutional_layer_create_info_t conv_default_config;
extern const layer_impl_t convolutional_layer_impl;
