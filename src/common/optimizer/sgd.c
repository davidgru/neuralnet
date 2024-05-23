#include "tensor/tensor_math.h"
#include "log.h"

#include "sgd.h"


typedef struct {
    float learning_rate;
    weight_regularizaton_kind_t weight_reg_kind;
    float weight_reg_strength;
} sgd_t;


static uint32_t sgd_init(void* private_data, const optimizer_config_t* create_info);
static uint32_t sgd_update_params(void* private_data, layer_param_ref_list_t* params);
static float sgd_get_learning_rate(void* private_data);
static void sgd_set_learning_rate(void* private_data, float learning_rate);

const optimizer_impl_t sgd_optimizer = {
    .init_func = sgd_init,
    .get_lr_func = sgd_get_learning_rate,
    .set_lr_func = sgd_set_learning_rate,
    .update_func = sgd_update_params,
    .deinit_func = NULL,
    .private_data_size = sizeof(sgd_t)
};


static uint32_t sgd_init(void* private_data, const optimizer_config_t* config)
{
    sgd_t* sgd = (sgd_t*)private_data;
    const sgd_config_t* sgd_config = (const sgd_config_t*)config;

    sgd->learning_rate = sgd_config->learning_rate;
    sgd->weight_reg_kind = sgd_config->weight_reg_kind;
    sgd->weight_reg_strength = sgd_config->weight_reg_strength;
}


static float sgd_get_learning_rate(void* private_data)
{
    sgd_t* sgd = (sgd_t*)private_data;
    
    return sgd->learning_rate;
}


static void sgd_set_learning_rate(void* private_data, float learning_rate)
{
    sgd_t* sgd = (sgd_t*)private_data;
    sgd->learning_rate = learning_rate;
}


static uint32_t sgd_update_params(void* private_data, layer_param_ref_list_t* params)
{
    sgd_t* sgd = (sgd_t*)private_data;
    
    for (size_t i = 0; i < params->num_params; i++) {
        tensor_t* param = params->param_refs[i].param;
        tensor_t* gradient = params->param_refs[i].gradient;
    
        /* regularization step */
        switch (sgd->weight_reg_kind) {
            case WEIGHT_REG_NONE:
            {
                /* No regularization to be applied. */
                break;
            }
            case WEIGHT_REG_L1:
                LOG_ERROR("sgd: unsupported weight reg kind\n");
                return 1;
            case WEIGHT_REG_L2:
            {
                /* params -= learning_rate * weight_reg_strength * 2 * params */
                tensor_scaled_add(param, param, -1.0f * sgd->learning_rate * 2.0f * sgd->weight_reg_strength);
                break;
            }
            default:
                LOG_ERROR("sgd: unknown weight reg kind\n");
                return 1;
        }

        /* main update step params -= learning_rate * gradient */
        tensor_scaled_add(param, gradient, -1.0f * sgd->learning_rate);
    }
}
