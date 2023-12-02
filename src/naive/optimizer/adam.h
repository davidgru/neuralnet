#pragma once


#include "core/optimizer.h"


typedef struct {
    float learning_rate;
    float gamma1; /* momentum factor for the mean square norm */
    float gamma2; /* momentum factor for the gradients */
    weight_regularizaton_kind_t weight_reg_kind;
    float weight_reg_strength;
} adam_config_t;


const extern optimizer_impl_t adam_optimizer;
