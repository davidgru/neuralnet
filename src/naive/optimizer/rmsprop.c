#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "log.h"
#include "util/ai_math.h"

#include "rmsprop.h"


typedef struct {
    float learning_rate;
    float gamma;
    weight_regularizaton_kind_t weight_reg_kind;
    float weight_reg_strength;
    tensor_t* running_gradient_mean_square;
    tensor_t* gradient_scratch;
} rmsprop_t;


static uint32_t rmsprop_init(void* private_data, const optimizer_config_t* create_info);
static float rmsprop_get_learning_rate(void* private_data);
static void rmsprop_set_learning_rate(void* private_data, float learning_rate);
static uint32_t rmsprop_update_params(void* private_data, layer_param_ref_list_t* params);


/* For numerical stability */
#define EPS 1e-8f


/* v = gamma * v + (1 - gamma) * gradient  */
static void mean_square_update(float* v, float* g, float gamma, size_t size);
static void param_update_step(float* params, float* v, float* g, float lr, size_t size);


const optimizer_impl_t rmsprop_optimizer = {
    .init_func = rmsprop_init,
    .get_lr_func = rmsprop_get_learning_rate,
    .set_lr_func = rmsprop_set_learning_rate,
    .update_func = rmsprop_update_params,
    .deinit_func = NULL,
    .private_data_size = sizeof(rmsprop_t)
};


static uint32_t rmsprop_init(void* private_data, const optimizer_config_t* config)
{
    rmsprop_t* rmsprop = (rmsprop_t*)private_data;
    const rmsprop_config_t* rmsprop_config = (const rmsprop_config_t*)config;

    rmsprop->learning_rate = rmsprop_config->learning_rate;
    rmsprop->gamma = rmsprop_config->gamma;
    rmsprop->weight_reg_kind = rmsprop_config->weight_reg_kind;
    rmsprop->weight_reg_strength = rmsprop_config->weight_reg_strength;
    rmsprop->running_gradient_mean_square = NULL;
    rmsprop->gradient_scratch = NULL;
}


static float rmsprop_get_learning_rate(void* private_data)
{
    rmsprop_t* rmsprop = (rmsprop_t*)private_data;
    
    return rmsprop->learning_rate;
}


static void rmsprop_set_learning_rate(void* private_data, float learning_rate)
{
    rmsprop_t* rmsprop = (rmsprop_t*)private_data;
    rmsprop->learning_rate = learning_rate;
}



static uint32_t rmsprop_update_params(void* private_data, layer_param_ref_list_t* params)
{
    rmsprop_t* rmsprop = (rmsprop_t*)private_data;
    
    /* allocate memory for the running gradients only when the function is called for the first
        time. init running radient norms with 0 */
    if (rmsprop->running_gradient_mean_square == NULL) {
        rmsprop->running_gradient_mean_square = (tensor_t*)calloc(params->num_params, sizeof(tensor_t));
        rmsprop->gradient_scratch = (tensor_t*)calloc(params->num_params, sizeof(tensor_t));
        for (size_t i = 0; i < params->num_params; i++) {
            const tensor_shape_t* gradient_shape = tensor_get_shape(params->param_refs[i].gradient);
            tensor_allocate(&rmsprop->running_gradient_mean_square[i], gradient_shape);
            tensor_set_zero(&rmsprop->running_gradient_mean_square[i]);
            tensor_allocate(&rmsprop->gradient_scratch[i], gradient_shape);
        }
    }


    for (size_t i = 0; i < params->num_params; i++) {
        tensor_t* param = params->param_refs[i].param;
        tensor_t* gradient = params->param_refs[i].gradient;

        float* param_data = tensor_get_data(param);
        float* gradient_data = tensor_get_data(gradient);
        float* gradient_scratch = tensor_get_data(&rmsprop->gradient_scratch[i]);
        float* running_squared_mean = tensor_get_data(&rmsprop->running_gradient_mean_square[i]);
        size_t param_size = tensor_size_from_shape(tensor_get_shape(param));

        /* gradient_scratch = gradient + regularization */
        VectorCopy(gradient_scratch, gradient_data, param_size);
        switch (rmsprop->weight_reg_kind) {
            case WEIGHT_REG_NONE:
            {
                /* No regularization to be applied. */
                break;
            }
            case WEIGHT_REG_L1:
                LOG_ERROR("rmsprop: unsupported weight reg kind\n");
                return 1;
            case WEIGHT_REG_L2:
            {
                /* params -= learning_rate * weight_reg_strength * 2 * params */
                VectorScaledAdd(gradient_scratch, param_data,
                    2.0f * rmsprop->weight_reg_strength, param_size);
                break;
            }
            default:
                LOG_ERROR("rmsprop: unknown weight reg kind\n");
                return 1;
        }

        /* eltwise v = gamma * v + (1 - gamma) * g * g */
        mean_square_update(running_squared_mean, gradient_scratch, rmsprop->gamma, param_size);

        /* eltwise update step is w <- w - lr / sqrt(v + eps) * g */
        param_update_step(param_data, running_squared_mean, gradient_scratch, rmsprop->learning_rate, param_size);
    }
}


static void mean_square_update(float* v, float* g, float gamma, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        v[i] = gamma * v[i] + (1.0f - gamma) * g[i] * g[i];
    }
}

static void param_update_step(float* params, float* v, float* g, float lr, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        params[i] -= (lr / sqrtf(v[i] + EPS)) * g[i];
    }
}
