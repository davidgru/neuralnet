#include <math.h>

#include "tensor/tensor.h"
#include "tensor/tensor_impl.h"

#include "layer/batchnorm_layer.h"
#include "tensor/tensor_math.h"
#include "util/ai_math.h"
#include "batchnorm_layer_internal.h"

#include "log.h"
#include <stdio.h>

#define BN_PARAMS_GAMMA_IDX 0
#define BN_PARAMS_BETA_IDX 1
#define BN_NUM_PARAMS 2

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

    tensor_t reduce_tmp1;
    tensor_t reduce_tmp2;
    tensor_t reduce_tmp3;

    batchnorm_layer_create_info_t config;
    device_t device;
} batchnorm_context_t;


static uint32_t batchnorm_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape,
    device_t device
)
{
    batchnorm_context_t* bn_context = context;
    
    bn_context->config = *(const batchnorm_layer_create_info_t*)create_info;
    bn_context->device = device;

    /* allocate gradient memory. only two scalar parameters */
    tensor_shape_t params_shape = make_tensor_shape(1, input_shape->dims[TENSOR_CHANNEL_DIM]);
    tensor_allocate_device(&bn_context->gamma, &params_shape, device);
    tensor_allocate_device(&bn_context->beta, &params_shape, device);
    tensor_allocate_device(&bn_context->d_gamma, &params_shape, device);
    tensor_allocate_device(&bn_context->d_beta, &params_shape, device);

    bn_context->param_refs[BN_PARAMS_GAMMA_IDX].param = &bn_context->gamma;
    bn_context->param_refs[BN_PARAMS_GAMMA_IDX].gradient = &bn_context->d_gamma;
    bn_context->param_refs[BN_PARAMS_BETA_IDX].param = &bn_context->beta;
    bn_context->param_refs[BN_PARAMS_BETA_IDX].gradient = &bn_context->d_beta;

    /* also need memory for mean and variance for each singular value in the activation */
    tensor_allocate_device(&bn_context->mean, &params_shape, device);
    tensor_allocate_device(&bn_context->var, &params_shape, device);
    tensor_allocate_device(&bn_context->d_mean, &params_shape, device);
    tensor_allocate_device(&bn_context->d_var, &params_shape, device);
    tensor_allocate_device(&bn_context->running_mean, &params_shape, device);
    tensor_allocate_device(&bn_context->running_var, &params_shape, device);

    tensor_shape_t reduce_tmp_shape = make_tensor_shape(2,
        input_shape->dims[TENSOR_BATCH_DIM],
        input_shape->dims[TENSOR_CHANNEL_DIM]
    );
    tensor_allocate_device(&bn_context->reduce_tmp1, &reduce_tmp_shape, device);
    tensor_allocate_device(&bn_context->reduce_tmp2, &reduce_tmp_shape, device);
    tensor_allocate_device(&bn_context->reduce_tmp3, input_shape, device);

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

    tensor_destory(&bn_context->reduce_tmp1);
    tensor_destory(&bn_context->reduce_tmp2);
    tensor_destory(&bn_context->reduce_tmp3);

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
    const size_t per_batch_size = num_channels * per_channel_size;

    const tensor_t* use_mean;
    const tensor_t* use_var;

    if (forward_kind == LAYER_FORWARD_TRAINING) {
        /* Determine per-value mean and variance along input batch. */

        const tensor_t input_view = {
            .shape = make_tensor_shape(3,
                tensor_batch_size(input),
                tensor_channels(input),
                tensor_per_channel_size(input)
            ),
            .device = bn_context->device,
            .data = input->data
        };

        /* mean */
        tensor_mean_axis(&bn_context->reduce_tmp1, &input_view, TENSOR_HEIGHT_DIM);
        tensor_mean_axis(&bn_context->mean, &bn_context->reduce_tmp1, TENSOR_BATCH_DIM);

        /* variance */
        /* need to broadcast mean to each batch manually to compute the variance - can be optimized */
        for (size_t i = 0; i < tensor_batch_size(input); i++) {
            tensor_t channel_means_view = {
                .shape = make_tensor_shape(1, tensor_channels(input)),
                .device = bn_context->device,
                .data = bn_context->reduce_tmp1.data + i * tensor_channels(input)
            };
            tensor_copy(&channel_means_view, &bn_context->mean);
        }
        tensor_variance_axis(&bn_context->reduce_tmp2, &input_view, &bn_context->reduce_tmp1, TENSOR_HEIGHT_DIM);
        tensor_mean_axis(&bn_context->var, &bn_context->reduce_tmp2, TENSOR_BATCH_DIM);

        /* update moving averages */
        tensor_momentum_update(&bn_context->running_mean, &bn_context->mean, bn_context->config.momentum);
        tensor_momentum_update(&bn_context->running_var, &bn_context->var, bn_context->config.momentum);

        use_mean = &bn_context->mean;
        use_var = &bn_context->var;
    } else {
        /* use moving averages to normalize the input during inference */
        use_mean = &bn_context->running_mean;
        use_var = &bn_context->running_var;
    }

    if (bn_context->device == device_cpu) {
        batchnorm_forward_cpu(input, use_mean, use_var, &bn_context->gamma,
            &bn_context->beta, out_output, bn_context->config.eps);
    }
#if defined(USE_GPU)
    else if (bn_context->device == device_gpu){
        batchnorm_forward_gpu(input, use_mean, use_var, &bn_context->gamma,
            &bn_context->beta, out_output, bn_context->config.eps);
    
    }
#endif
    else {
        LOG_ERROR("Invalid device.\n");
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

    if (bn_context->device == device_cpu) {
        batchnorm_backward_data_cpu(input, &bn_context->mean, &bn_context->var, &bn_context->gamma,
            prev_gradient, &bn_context->d_mean, &bn_context->d_var, out_gradient, bn_context->config.eps);
        batchnorm_backward_weights_cpu(input, &bn_context->mean, &bn_context->var, prev_gradient,
            &bn_context->d_gamma, &bn_context->d_beta, bn_context->config.eps);
    }
#if defined(USE_GPU)
    else if (bn_context->device == device_gpu) {
        batchnorm_backward_data_gpu(input, &bn_context->mean, &bn_context->var, &bn_context->gamma,
            prev_gradient, &bn_context->d_mean, &bn_context->d_var, out_gradient, bn_context->config.eps,
            &bn_context->reduce_tmp1, &bn_context->reduce_tmp3);
        batchnorm_backward_weights_gpu(input, &bn_context->mean, &bn_context->var, prev_gradient,
            &bn_context->d_gamma, &bn_context->d_beta, bn_context->config.eps,
            &bn_context->reduce_tmp1, &bn_context->reduce_tmp3);
    
    }
#endif
    else {
        LOG_ERROR("Device not supported\n");
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


const layer_impl_t batchnorm_layer_impl = {
    .init_func = batchnorm_layer_init,
    .get_param_func = batchnorm_layer_get_params,
    .deinit_func = batchnorm_layer_deinit,
    .forward_func = batchnorm_layer_forward,
    .backward_func = batchnorm_layer_backward,
    .calc_output_size = batchnorm_layer_calc_output_shape,
    .layer_context_size = sizeof(batchnorm_context_t),
};
