#pragma once

#include "core/layer.h"
#include "core/layer_impl.h"

#include "util/weight_init.h"


typedef struct {
    size_t output_size;
    weight_init_func_t weight_init;
    weight_init_func_t bias_init;
} linear_layer_create_info_t;


extern const layer_impl_t linear_layer_impl;
