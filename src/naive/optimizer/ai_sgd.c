#include "log.h"
#include "util/ai_math.h"

#include "ai_sgd.h"


typedef struct {
    float learning_rate;
    sgd_weight_regularizaton_kind_t weight_reg_kind;
    float weight_reg_strength;
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
    sgd->weight_reg_kind = sgd_config->weight_reg_kind;
    sgd->weight_reg_strength = sgd_config->weight_reg_strength;
}


static uint32_t sgd_update_params(void* private_data, layer_param_ref_list_t* params)
{
    sgd_t* sgd = (sgd_t*)private_data;
    
    for (size_t i = 0; i < params->num_params; i++) {
        tensor_t* param = params->param_refs[i].param;
        tensor_t* gradient = params->param_refs[i].gradient;

        float* param_data = tensor_get_data(param);
        float* gradient_data = tensor_get_data(gradient);
        size_t param_size = tensor_size_from_shape(tensor_get_shape(param));
        
        /* regularization step */
        switch (sgd->weight_reg_kind) {
            case SGD_WEIGHT_REG_NONE:
            {
                /* No regularization to be applied. */
                break;
            }
            case SGD_WEIGHT_REG_L1:
                LOG_ERROR("sgd: unsupported weight reg kind\n");
                return 1;
            case SGD_WEIGHT_REG_L2:
            {
                /* params -= learning_rate * weight_reg_strength * 2 * params */
                AI_VectorScaledAdd(param_data, param_data,
                    (-1.0f) * sgd->learning_rate * 2.0f * sgd->weight_reg_strength, param_size);
                break;
            }
            default:
                LOG_ERROR("sgd: unknown weight reg kind\n");
                return 1;
        }

        /* main update step params -= learning_rate * gradient */
        AI_VectorScaledAdd(param_data, gradient_data, (-1.0f) * sgd->learning_rate, param_size);
    }
}
