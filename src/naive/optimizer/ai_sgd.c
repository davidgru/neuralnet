#include "ai_sgd.h"


#include "util/ai_math.h"


typedef struct {
    float learning_rate;
} sgd_t;


static uint32_t sgd_init(void* private_data, const optimizer_config_t* create_info);
static uint32_t sgd_update_params(void* private_data, layer_param_ref_list_t* params);


const optimizer_impl_t sgd_optimizer = {
    .init_func = sgd_init,
    .update_func = sgd_update_params,
    .deinit_func = NULL,
    .private_data_size = sizeof(sgd_t)
};


static uint32_t sgd_init(void* private_data, const optimizer_config_t* config)
{
    sgd_t* sgd = (sgd_t*)private_data;
    const sgd_config_t* sgd_config = (const sgd_config_t*)config;

    sgd->learning_rate = sgd_config->learning_rate;
}


static uint32_t sgd_update_params(void* private_data, layer_param_ref_list_t* params)
{
    sgd_t* sgd = (sgd_t*)private_data;
    
    for (size_t i = 0; i < params->num_params; i++) {
        tensor_t* param = params->param_refs[i].param;
        tensor_t* gradient = params->param_refs[i].gradient;
        AI_VectorScaledAdd(tensor_get_data(param), tensor_get_data(gradient),
            (-1.0) * sgd->learning_rate, tensor_size_from_shape(tensor_get_shape(param)));
    }
}
