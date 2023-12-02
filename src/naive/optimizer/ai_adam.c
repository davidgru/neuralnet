#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "log.h"
#include "util/ai_math.h"

#include "ai_adam.h"


typedef struct {
    float learning_rate;
    float gamma1;
    float gamma2;
    weight_regularizaton_kind_t weight_reg_kind;
    float weight_reg_strength;
    tensor_t* running_gradient_mean_square;
    tensor_t* running_gradient;
    tensor_t* gradient_scratch;
    size_t iter;
} adam_t;


static uint32_t adam_init(void* private_data, const optimizer_config_t* create_info);
static float adam_get_learning_rate(void* private_data);
static void adam_set_learning_rate(void* private_data, float learning_rate);
static uint32_t adam_update_params(void* private_data, layer_param_ref_list_t* params);


/* For numerical stability */
#define EPS 1e-8f


/* v = gamma * v + (1 - gamma) * gradient  */
static void mean_square_update(float* v, float* g, float gamma, size_t size);
static void param_update_step(adam_t* adam, float* params, float* v, float* m, size_t size);


const optimizer_impl_t adam_optimizer = {
    .init_func = adam_init,
    .get_lr_func = adam_get_learning_rate,
    .set_lr_func = adam_set_learning_rate,
    .update_func = adam_update_params,
    .deinit_func = NULL,
    .private_data_size = sizeof(adam_t)
};


static uint32_t adam_init(void* private_data, const optimizer_config_t* config)
{
    adam_t* adam = (adam_t*)private_data;
    const adam_config_t* adam_config = (const adam_config_t*)config;

    adam->learning_rate = adam_config->learning_rate;
    adam->gamma1 = adam_config->gamma1;
    adam->gamma2 = adam_config->gamma2;
    adam->weight_reg_kind = adam_config->weight_reg_kind;
    adam->weight_reg_strength = adam_config->weight_reg_strength;
    adam->running_gradient_mean_square = NULL;
    adam->gradient_scratch = NULL;
    adam->iter = 0;
}


static float adam_get_learning_rate(void* private_data)
{
    adam_t* adam = (adam_t*)private_data;
    
    return adam->learning_rate;
}


static void adam_set_learning_rate(void* private_data, float learning_rate)
{
    adam_t* adam = (adam_t*)private_data;
    adam->learning_rate = learning_rate;
}


static uint32_t adam_update_params(void* private_data, layer_param_ref_list_t* params)
{
    adam_t* adam = (adam_t*)private_data;
    
    /* allocate memory for the running gradients only when the function is called for the first
        time. init running radient norms with 0 */
    if (adam->running_gradient_mean_square == NULL) {
        adam->running_gradient_mean_square = (tensor_t*)calloc(params->num_params, sizeof(tensor_t));
        adam->running_gradient = (tensor_t*)calloc(params->num_params, sizeof(tensor_t));
        adam->gradient_scratch = (tensor_t*)calloc(params->num_params, sizeof(tensor_t));
        for (size_t i = 0; i < params->num_params; i++) {
            const tensor_shape_t* gradient_shape = tensor_get_shape(params->param_refs[i].gradient);
            tensor_allocate(&adam->running_gradient_mean_square[i], gradient_shape);
            tensor_allocate(&adam->running_gradient[i], gradient_shape);
            tensor_allocate(&adam->gradient_scratch[i], gradient_shape);
            tensor_set_zero(&adam->running_gradient_mean_square[i]);
            tensor_set_zero(&adam->running_gradient[i]);
        }
    }
    adam->iter++;


    for (size_t i = 0; i < params->num_params; i++) {
        tensor_t* param = params->param_refs[i].param;
        tensor_t* gradient = params->param_refs[i].gradient;

        float* param_data = tensor_get_data(param);
        float* gradient_data = tensor_get_data(gradient);
        float* gradient_scratch = tensor_get_data(&adam->gradient_scratch[i]);
        float* running_squared_mean = tensor_get_data(&adam->running_gradient_mean_square[i]);
        float* running_gradient = tensor_get_data(&adam->running_gradient[i]);
        size_t param_size = tensor_size_from_shape(tensor_get_shape(param));

        /* gradient_scratch = gradient + regularization */
        AI_VectorCopy(gradient_scratch, gradient_data, param_size);
        switch (adam->weight_reg_kind) {
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
                AI_VectorScaledAdd(gradient_scratch, param_data,
                    2.0f * adam->weight_reg_strength, param_size);
                break;
            }
            default:
                LOG_ERROR("rmsprop: unknown weight reg kind\n");
                return 1;
        }

        /* gradient norm update: eltwise v <- gamma1 * v + (1 - gamma1) * g * g */
        mean_square_update(running_squared_mean, gradient_scratch, adam->gamma1, param_size);

        /* gradient momentum update: m <- gamma2 * m + (1 - gamma2) * g */
        AI_VectorScale(running_gradient, adam->gamma2, param_size);
        AI_VectorScaledAdd(running_gradient, gradient_scratch, 1.0f - adam->gamma2, param_size);
        
        /* eltwise update step is w <- w - (lr * c2) / (c1 * sqrt(v + eps)) * g */
        param_update_step(adam, param_data, running_squared_mean, running_gradient, param_size);
    }
}


static void mean_square_update(float* v, float* g, float gamma, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        v[i] = gamma * v[i] + (1.0f - gamma) * g[i] * g[i];
    }
}

static void param_update_step(adam_t* adam, float* params, float* v, float* m, size_t size)
{
    /* normalizers to fix underestimation */
    float c1 = 1.0f / (1.0f - powf(adam->gamma1, adam->iter));
    float c2 = 1.0f / (1.0f - powf(adam->gamma2, adam->iter));
        
    for (size_t i = 0; i < size; i++) {
        params[i] -= ((adam->learning_rate * c2) / sqrtf(c1 * v[i] + EPS)) * m[i];
        // params[i] -= adam->learning_rate * m[i];
    }
}
