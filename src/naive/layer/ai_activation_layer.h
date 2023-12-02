#pragma once


#include "ai_layer.h"


typedef enum {
    AI_ACTIVATION_FUNCTION_SIGMOID,
    AI_ACTIVATION_FUNCTION_TANH,
    AI_ACTIVATION_FUNCTION_RELU,
    AI_ACTIVATION_FUNCTION_LEAKY_RELU,
    /* AI_ACTIVATION_FUNCTION_SOFTMAX, */
} activation_function_kind_t;

typedef struct {
    activation_function_kind_t activation_function;
} activation_layer_create_info_t;


extern const layer_impl_t activation_layer_impl;
