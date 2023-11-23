#pragma once


#include "ai_optimizer.h"


typedef struct {
    float learning_rate;
} sgd_config_t;


const extern optimizer_impl_t sgd_optimizer;
