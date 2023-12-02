#pragma once

#include "layer/ai_layer.h"

#include "util/ai_weight_init.h"


typedef struct {
    size_t output_size;
    AI_FCLayerWeightInit weight_init;
    AI_FCLayerBiasInit bias_init;
} linear_layer_create_info_t;


extern const layer_impl_t linear_layer_impl;
