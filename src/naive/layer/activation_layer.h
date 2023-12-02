#pragma once


#include "core/layer.h"


typedef enum {
    ACTIVATION_FUNCTION_SIGMOID,
    ACTIVATION_FUNCTION_TANH,
    ACTIVATION_FUNCTION_RELU,
    ACTIVATION_FUNCTION_LEAKY_RELU,
} activation_function_kind_t;


typedef struct {
    activation_function_kind_t activation_function;
} activation_layer_create_info_t;


extern const layer_impl_t activation_layer_impl;
