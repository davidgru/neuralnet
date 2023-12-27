#pragma once

#include "core/layer.h"
#include "core/layer_impl.h"


typedef struct {
    float momentum; /* momentum in mean and variance ma */
    float eps; /* used for stability when dividing by variance */
} batchnorm_layer_create_info_t;


extern const batchnorm_layer_create_info_t batchnorm_default_config;
extern const layer_impl_t batchnorm_layer_impl;
