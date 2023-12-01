#pragma once

#include "ai_layer.h"


typedef struct {
    size_t output_channels;
    size_t filter_height;
    size_t filter_width;
    size_t stride_y;
    size_t stride_x;
    AI_ConvLayerWeightInit weight_init;
    AI_ConvLayerBiasInit bias_init;
} convolutional_layer_create_info_t;


extern const layer_info_t convolutional_layer_info;
