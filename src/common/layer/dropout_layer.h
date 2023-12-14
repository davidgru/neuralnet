#pragma once


#include "core/layer.h"
#include "core/layer_impl.h"


typedef struct {
    float dropout_rate;
} dropout_layer_create_info_t;


extern const layer_impl_t dropout_layer_impl;
