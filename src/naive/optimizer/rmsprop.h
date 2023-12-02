#pragma once


#include "core/optimizer.h"


typedef struct {
    float learning_rate;
    float gamma; /* momentum factor for the gradients */
    weight_regularizaton_kind_t weight_reg_kind;
    float weight_reg_strength;
} rmsprop_config_t;


const extern optimizer_impl_t rmsprop_optimizer;
