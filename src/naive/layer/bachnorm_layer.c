#include <math.h>

#include "tensor.h"
#include "tensor_impl.h"

#include "layer/bachnorm_layer.h"
#include "util/ai_math.h"

#include "log.h"
#include <stdio.h>

#define BN_PARAMS_GAMMA_IDX 0
#define BN_PARAMS_BETA_IDX 1
#define BN_NUM_PARAMS 2

#define BN_UPDATE_MOMENTUM 0.9f
#define BN_EPS 1e-8f


typedef struct {
    tensor_t gamma; /* scale */
    tensor_t beta; /* shift */
    tensor_t d_gamma;
    tensor_t d_beta;
    layer_param_ref_t param_refs[BN_NUM_PARAMS];

    tensor_t mean;
    tensor_t var;
    tensor_t d_mean;
    tensor_t d_var;

    tensor_t running_mean;
    tensor_t running_var;
} batchnorm_context_t;


static uint32_t batchnorm_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
);

static uint32_t batchnorm_layer_get_params(
    layer_context_t* context,
    layer_param_ref_list_t* out_layer_params
);

static uint32_t batchnorm_layer_deinit(layer_context_t* context);

static uint32_t batchnorm_layer_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t* out_output
);

static uint32_t batchnorm_layer_backward(
    layer_context_t* context,
    const tensor_t* input,
    const tensor_t* output,
    const tensor_t* prev_gradient,
    tensor_t* out_gradient
);

static uint32_t batchnorm_layer_calc_output_shape(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
);


const layer_impl_t batchnorm_layer_impl = {
    .init_func = batchnorm_layer_init,
    .get_param_func = batchnorm_layer_get_params,
    .deinit_func = batchnorm_layer_deinit,
    .forward_func = batchnorm_layer_forward,
    .backward_func = batchnorm_layer_backward,
    .calc_output_size = batchnorm_layer_calc_output_shape,
    .layer_context_size = sizeof(batchnorm_context_t),
};


static uint32_t batchnorm_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
)
{
    batchnorm_context_t* bn_context = context;


    /* allocate gradient memory. only two scalar parameters */
    tensor_shape_t params_shape = {.dims = { 0, 0, 0, input_shape->dims[TENSOR_CHANNEL_DIM]}};
    tensor_allocate(&bn_context->gamma, &params_shape);
    tensor_allocate(&bn_context->beta, &params_shape);
    tensor_allocate(&bn_context->d_gamma, &params_shape);
    tensor_allocate(&bn_context->d_beta, &params_shape);

    bn_context->param_refs[BN_PARAMS_GAMMA_IDX].param = &bn_context->gamma;
    bn_context->param_refs[BN_PARAMS_GAMMA_IDX].gradient = &bn_context->d_gamma;
    bn_context->param_refs[BN_PARAMS_BETA_IDX].param = &bn_context->beta;
    bn_context->param_refs[BN_PARAMS_BETA_IDX].gradient = &bn_context->d_beta;

    /* also need memory for mean and variance for each singular value in the activation */
    tensor_allocate(&bn_context->mean, &params_shape);
    tensor_allocate(&bn_context->var, &params_shape);
    tensor_allocate(&bn_context->d_mean, &params_shape);
    tensor_allocate(&bn_context->d_var, &params_shape);
    tensor_allocate(&bn_context->running_mean, &params_shape);
    tensor_allocate(&bn_context->running_var, &params_shape);
    

    /* init params. scale <- 1.0f, offset <- 0.0f */
    tensor_fill(&bn_context->gamma, 1.0f);
    tensor_fill(&bn_context->beta, 0.0f);

    /* mean <- 0.0f, var <- 1.0f */
    tensor_fill(&bn_context->mean, 0.0f);
    tensor_fill(&bn_context->running_mean, 0.0f);
    tensor_fill(&bn_context->var, 1.0f);
    tensor_fill(&bn_context->running_var, 1.0f);

    return 0;
}


static uint32_t batchnorm_layer_get_params(
    layer_context_t* context,
    layer_param_ref_list_t* out_layer_params
)
{
    batchnorm_context_t* bn_context = context;

    out_layer_params->param_refs = bn_context->param_refs;
    out_layer_params->num_params = BN_NUM_PARAMS;

    return 0;
}


static uint32_t batchnorm_layer_deinit(layer_context_t* context)
{
    batchnorm_context_t* bn_context = context;

    tensor_destory(&bn_context->gamma);
    tensor_destory(&bn_context->d_gamma);
    tensor_destory(&bn_context->beta);
    tensor_destory(&bn_context->d_beta);

    tensor_destory(&bn_context->mean);
    tensor_destory(&bn_context->var);
    tensor_destory(&bn_context->d_mean);
    tensor_destory(&bn_context->d_var);
    tensor_destory(&bn_context->running_mean);
    tensor_destory(&bn_context->running_var);

    return 0;
}


static uint32_t batchnorm_layer_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t* out_output
)
{
    batchnorm_context_t* bn_context = context;


    const float* input_buf = tensor_get_data_const(input);
    const tensor_shape_t* shape = tensor_get_shape(input);
    const size_t per_channel_size = shape->dims[TENSOR_HEIGHT_DIM] * shape->dims[TENSOR_WIDTH_DIM];
    const size_t num_channels = shape->dims[TENSOR_CHANNEL_DIM];
    const size_t batch_size = shape->dims[TENSOR_BATCH_DIM];
    const size_t per_batch_size = per_channel_size * num_channels;


    float* output_buf = tensor_get_data(out_output);
    float* running_mean_buf = tensor_get_data(&bn_context->running_mean);
    float* running_var_buf = tensor_get_data(&bn_context->running_var);


    if (forward_kind == LAYER_FORWARD_TRAINING) {
        /* Determine per-value mean and variance along input batch. */

        /* Determine mean first */
        float* mean_buf = tensor_get_data(&bn_context->mean);
        for (size_t ch = 0; ch < num_channels; ch++) {
            float acc = 0.0f;
            for (size_t n = 0; n < batch_size; n++) {
                for (size_t i = 0; i < per_channel_size; i++) {
                    acc += input_buf[n * per_batch_size + ch * per_channel_size + i];
                }
            }
            mean_buf[ch] = acc / (float)(batch_size * per_channel_size);
        }

        /* Determine variance given mean */
        float* var_buf = tensor_get_data(&bn_context->var);
        for (size_t ch = 0; ch < num_channels; ch++) {
            float acc = 0.0f;
            for (size_t n = 0; n < batch_size; n++) {
                for (size_t i = 0; i < per_channel_size; i++) {
                    float s = input_buf[n * per_batch_size + ch * per_channel_size + i]
                        - mean_buf[ch];
                    acc += s * s;
                }
            }
            var_buf[ch] = acc / (float)(batch_size * per_channel_size);
        }

        /* Apply normalization to obtain \hat{x} */
        for (size_t n = 0; n < batch_size; n++) {
            for (size_t ch = 0; ch < num_channels; ch++) {
                for (size_t i = 0; i < per_channel_size; i++) {
                    output_buf[n * per_batch_size + ch * per_channel_size + i] =
                        (input_buf[n * per_batch_size + ch * per_channel_size + i] - mean_buf[ch])
                        / sqrtf(var_buf[ch] + BN_EPS);
                }
            }
        }

        /* update moving averages */
        VectorScale(running_mean_buf, BN_UPDATE_MOMENTUM, num_channels);
        VectorScale(running_var_buf, BN_UPDATE_MOMENTUM, num_channels);
        VectorScaledAdd(running_mean_buf, mean_buf, 1.0f - BN_UPDATE_MOMENTUM, num_channels);
        VectorScaledAdd(running_var_buf, var_buf, 1.0f - BN_UPDATE_MOMENTUM, num_channels);
    } else {
        /* use moving averages to normalize the input during inference */
        for (size_t n = 0; n < batch_size; n++) {
            for (size_t ch = 0; ch < num_channels; ch++) {
                for (size_t i = 0; i < per_channel_size; i++) {
                    output_buf[n * per_batch_size + ch * per_channel_size + i] =
                        (input_buf[n * per_batch_size + ch * per_channel_size + i]
                        - running_mean_buf[ch]) / sqrtf(running_var_buf[ch] + BN_EPS);
                }
            }
        }
    }


    /* apply scale and shift */
    const float* gamma_buf = tensor_get_data_const(&bn_context->gamma);
    const float* beta_buf = tensor_get_data_const(&bn_context->beta);
    for (size_t n = 0; n < batch_size; n++) {
        for (size_t ch = 0; ch < num_channels; ch++) {
            for (size_t i = 0; i < per_channel_size; i++) {
                const size_t idx = n * per_batch_size + ch * per_channel_size + i;
                output_buf[idx] = gamma_buf[ch] * output_buf[idx] + beta_buf[ch];
            }
        }
    }

    return 0;
}


static uint32_t batchnorm_layer_backward(
    layer_context_t* context,
    const tensor_t* input,
    const tensor_t* output,
    const tensor_t* prev_gradient,
    tensor_t* out_gradient
)
{
    batchnorm_context_t* bn_context = context;

    const tensor_shape_t* shape = tensor_get_shape(input);
    const size_t per_channel_size = shape->dims[TENSOR_HEIGHT_DIM] * shape->dims[TENSOR_WIDTH_DIM];
    const size_t num_channels = shape->dims[TENSOR_CHANNEL_DIM];
    const size_t batch_size = shape->dims[TENSOR_BATCH_DIM];
    const size_t per_batch_size = per_channel_size * num_channels;

    const float* input_buf = tensor_get_data_const(input);
    const float* gamma_buf = tensor_get_data_const(&bn_context->gamma);

    const float* mean_buf = tensor_get_data_const(&bn_context->mean);
    const float* var_buf = tensor_get_data_const(&bn_context->var);

    float* d_mean_buf = tensor_get_data(&bn_context->d_mean);
    float* d_var_buf = tensor_get_data(&bn_context->d_var);
    float* d_x_buf = tensor_get_data(out_gradient);
    const float* x_buf = tensor_get_data_const(input);
    const float* d_y_buf = tensor_get_data_const(prev_gradient);

    /* calculate d_\hat{x} = d_y * gamma. can make use of d_x memory */
    tensor_set_zero(out_gradient);
    for (size_t n = 0; n < batch_size; n++) {
        for (size_t ch = 0; ch < num_channels; ch++) {
            const size_t off = n * per_batch_size + ch * per_channel_size;
            VectorScaledAdd(&d_x_buf[off], &d_y_buf[off], gamma_buf[ch], per_channel_size);
        }
    }

    /* calculate d_var */
    for (size_t ch = 0; ch < num_channels; ch++) {
        float acc = 0.0f;
        for (size_t n = 0; n < batch_size; n++) {
            for (size_t i = 0; i < per_channel_size; i++) {
                const size_t idx = n * per_batch_size + ch * per_channel_size + i;
                acc += d_x_buf[idx] * (x_buf[idx] - mean_buf[ch]);
            }
        }
        d_var_buf[ch] = acc * -0.5f * powf(var_buf[ch] + BN_EPS, -1.5f);
    }

    /* calculate d_mean */
    for (size_t ch = 0; ch < num_channels; ch++) {
        float acc1 = 0.0f; /* sum_{i=1..m} d\hat{x}_i */
        float acc2 = 0.0f; /* sum_{i=1..m} -2.0f * (x_i - mean) */
        for (size_t n = 0; n < batch_size; n++) {
            for (size_t i = 0; i < per_channel_size; i++) {
                const size_t idx = n * per_batch_size + ch * per_channel_size + i;
                acc1 += d_x_buf[idx];
                acc2 += -2.0f * (x_buf[n * per_batch_size + i] - mean_buf[ch]);
            }
        }
        d_mean_buf[ch] = -acc1 / sqrtf(var_buf[ch] + BN_EPS) + d_var_buf[ch] * acc2 / (float)(batch_size * per_channel_size);
    }

    /* calculate gradient of scale gamma and shift beta */
    float* d_gamma_buf = tensor_get_data(&bn_context->d_gamma);
    float* d_beta_buf = tensor_get_data(&bn_context->d_beta);
    for (size_t ch = 0; ch < num_channels; ch++) {
        float d_gamma_acc = 0.0f;
        float d_beta_acc = 0.0f;
        for (size_t n = 0; n < batch_size; n++) {
            for (size_t i = 0; i < per_channel_size; i++) {
                const size_t idx = n * per_batch_size + ch * per_channel_size + i;
                float x_hat = (x_buf[idx] - mean_buf[ch]) / sqrtf(var_buf[ch] + BN_EPS);
                d_gamma_acc += d_y_buf[idx] * x_hat;
                d_beta_acc += d_y_buf[idx];
            }
        }
        d_gamma_buf[ch] = d_gamma_acc / (float)(batch_size);
        d_beta_buf[ch] = d_beta_acc / (float)(batch_size);
    }

    /* d_\hat{x} not needed anymore */

    /* calculate d_x */
    for (size_t n = 0; n < batch_size; n++) {
        for (size_t ch = 0; ch < num_channels; ch++) {
            for (size_t i = 0; i < per_channel_size; i++) {
                const size_t idx = n * per_batch_size + ch * per_channel_size + i;
                d_x_buf[idx] = d_x_buf[idx] / sqrtf(var_buf[ch] + BN_EPS)
                    + d_var_buf[ch] * 2.0f * (x_buf[idx] - mean_buf[ch]) / (float)(batch_size * per_channel_size)
                    + d_mean_buf[ch] / (float)(batch_size * per_channel_size);
            }
        }
    }

    return 0;
}


static uint32_t batchnorm_layer_calc_output_shape(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
)
{
    /* Batchnorm leaves dimensions unchanged. */
    *out_output_shape = *input_shape;

    return 0;
}
