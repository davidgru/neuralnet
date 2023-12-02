#pragma once


#include "layer/ai_layer.h"


typedef void optimizer_config_t;


typedef enum {
    WEIGHT_REG_NONE,
    WEIGHT_REG_L1,
    WEIGHT_REG_L2
} weight_regularizaton_kind_t;



typedef uint32_t (*optimizer_init_func_t)(
    void* private_data,
    const optimizer_config_t* create_info
);


typedef float (*optimizer_get_learning_rate_func_t)(void* private_data);


typedef void (*optimizer_set_learning_rate_func_t)(
    void* private_data,
    float learning_rate
);


typedef uint32_t (*optimizer_update_params_func_t)(
    void* private_data, 
    layer_param_ref_list_t* params
);


typedef uint32_t (*optimizer_deinit_func_t)(void* private_data);


typedef struct {
    optimizer_init_func_t init_func;
    optimizer_get_learning_rate_func_t get_lr_func;
    optimizer_set_learning_rate_func_t set_lr_func;
    optimizer_update_params_func_t update_func;
    optimizer_deinit_func_t deinit_func;
    size_t private_data_size;
} optimizer_impl_t;


typedef struct optimizer_s* optimizer_t;


uint32_t optimizer_create(
    optimizer_t* optimizer,
    const optimizer_impl_t* impl,
    const optimizer_config_t* config
);


uint32_t optimizer_add_params(optimizer_t optimizer, layer_param_ref_list_t* refs);


float optimizer_get_learning_rate(optimizer_t optimizer);


void optimizer_set_learning_rate(optimizer_t optimizer, float learning_rate);


uint32_t optimizer_step(optimizer_t optimizer);


uint32_t optimizer_destroy(optimizer_t optimizer);
