#pragma once


#include "ai_base_layer.h"


void input_layer_init(AI_Layer** layer, void* create_info, AI_Layer* prev_layer);


uint32_t input_layer_calc_output_shape(
    tensor_shape_t* out_output_shape,
    const void* create_info,
    const tensor_shape_t* input_shape
);
