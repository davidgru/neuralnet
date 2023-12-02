#pragma once


#include "core/ai_layer.h"


typedef struct {
    float dropout_rate;
} dropout_layer_create_info_t;


extern const layer_impl_t dropout_layer_impl;
