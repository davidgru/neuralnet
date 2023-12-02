#pragma once


#include "core/ai_optimizer.h"


typedef enum {
    SGD_WEIGHT_REG_NONE,
    SGD_WEIGHT_REG_L1,
    SGD_WEIGHT_REG_L2
} sgd_weight_regularizaton_kind_t;


typedef struct {
    float learning_rate;
    sgd_weight_regularizaton_kind_t weight_reg_kind;
    float weight_reg_strength;
} sgd_config_t;


const extern optimizer_impl_t sgd_optimizer;
