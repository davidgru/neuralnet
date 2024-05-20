/* Based on: http://cs231n.stanford.edu/handouts/linear-backprop.pdf */
#include <malloc.h>
#include <string.h>

#include "util/ai_math.h"
#include "tensor_impl.h"
#include "tensor/tensor_math.h"
#include "log.h"

#include "layer/linear_layer.h"
#include "layer/linear/linear_layer_internal.h"


#define NUM_LINEAR_LAYER_PARAMS 2
#define LINEAR_LAYER_WEIGHTS_PARAM 0
#define LINEAR_LAYER_BIAS_PARAM 1


#define LINEAR_WEIGHTS_OUTPUT_DIM 0
#define LINEAR_WEIGHTS_INPUT_DIM  1


typedef struct linear_layer_t {
    tensor_t weights;
    tensor_t bias;
    tensor_t d_weights;
    tensor_t d_bias;
    layer_param_ref_t param_refs[NUM_LINEAR_LAYER_PARAMS];
    device_t device;
} linear_layer_t;


static uint32_t linear_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
);

static uint32_t linear_layer_get_params(
    layer_context_t* context,
    layer_param_ref_list_t* out_layer_params
);

static uint32_t linear_layer_deinit(layer_context_t* context);

static uint32_t linear_layer_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t* out_output
);

static uint32_t linear_layer_backward(
    layer_context_t* context,
    const tensor_t* input,
    const tensor_t* output,
    const tensor_t* prev_gradient,
    tensor_t* out_gradient
);

static uint32_t linear_layer_calc_output_shape(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
);


static uint32_t linear_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
)
{
    linear_layer_t* linear_layer = (linear_layer_t*)context;
    const linear_layer_create_info_t* linear_create_info = (linear_layer_create_info_t*)create_info;


    /* For now implicitly flatten input. Might be benefical to implement an flatten layer in
        future. */
    tensor_shape_t weights_shape = {
        .dims[LINEAR_WEIGHTS_OUTPUT_DIM] = output_shape->dims[TENSOR_CHANNEL_DIM],
        .dims[LINEAR_WEIGHTS_INPUT_DIM] = input_shape->dims[TENSOR_CHANNEL_DIM]
            * input_shape->dims[TENSOR_HEIGHT_DIM]
            * input_shape->dims[TENSOR_WIDTH_DIM],
        .dims[2] = 0,
        .dims[3] = 0,
    };
    tensor_allocate_device(&linear_layer->weights, &weights_shape, linear_layer->device);
    tensor_allocate_device(&linear_layer->d_weights, &weights_shape, linear_layer->device);

    tensor_shape_t bias_shape = {
        .dims[0] = linear_create_info->output_size,
        .dims[1] = 0,
        .dims[2] = 0,
        .dims[3] = 0,
    };
    tensor_allocate_device(&linear_layer->bias, &bias_shape, linear_layer->device);
    tensor_allocate_device(&linear_layer->d_bias, &bias_shape, linear_layer->device);

    /* need to register the params for the optimizer */
    linear_layer->param_refs[LINEAR_LAYER_WEIGHTS_PARAM].param = &linear_layer->weights;
    linear_layer->param_refs[LINEAR_LAYER_WEIGHTS_PARAM].gradient = &linear_layer->d_weights;
    linear_layer->param_refs[LINEAR_LAYER_BIAS_PARAM].param = &linear_layer->bias;
    linear_layer->param_refs[LINEAR_LAYER_BIAS_PARAM].gradient = &linear_layer->d_bias;


    /* Initialise weights and bias */
    if (linear_layer->device == device_gpu) {
        tensor_t tmp_weights;
        tensor_t tmp_bias;
        tensor_allocate_device(&tmp_weights, &weights_shape, device_cpu);
        tensor_allocate_device(&tmp_bias, &bias_shape, device_cpu);
        linear_create_info->weight_init(&tmp_weights);
        linear_create_info->bias_init(&tmp_bias);
        tensor_copy(&linear_layer->weights, &tmp_weights);
        tensor_copy(&linear_layer->bias, &tmp_bias);
        tensor_destory(&tmp_weights);
        tensor_destory(&tmp_bias);
    } else {
        linear_create_info->weight_init(&linear_layer->weights);
        linear_create_info->bias_init(&linear_layer->bias);
    }

    return 0;
};


static uint32_t linear_layer_get_params(
    layer_context_t* context,
    layer_param_ref_list_t* out_layer_params
)
{
    linear_layer_t* linear_layer = (linear_layer_t*)context;

    out_layer_params->param_refs = linear_layer->param_refs;
    out_layer_params->num_params = NUM_LINEAR_LAYER_PARAMS;
    return 0;
}


static uint32_t linear_layer_deinit(layer_context_t* context)
{
    linear_layer_t* linear_layer = (linear_layer_t*)context;
    
    tensor_destory(&linear_layer->weights);
    tensor_destory(&linear_layer->d_weights);
    tensor_destory(&linear_layer->bias);
    tensor_destory(&linear_layer->d_bias);
}


static uint32_t linear_layer_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t* out_output
)
{
    linear_layer_t* linear_layer = (linear_layer_t*)context;


    const tensor_shape_t* input_shape = tensor_get_shape(input);
    const tensor_shape_t* output_shape = tensor_get_shape(out_output);


    const float* input_data = tensor_get_data_const(input);
    const float* weights = tensor_get_data_const(&linear_layer->weights);
    float* output_data = tensor_get_data(out_output);


    const size_t batch_size = input_shape->dims[TENSOR_BATCH_DIM];
    const size_t per_batch_input_size = tensor_size_from_shape(input_shape) / batch_size;
    const size_t output_size = output_shape->dims[TENSOR_CHANNEL_DIM];


    /* output = input * weights */
    matrix_product(input_data, weights, output_data, batch_size, output_size, per_batch_input_size);

    /* output += bias */
    for (size_t i = 0; i < batch_size; i++) {
        tensor_t curr_batch = {
            .shape = make_tensor_shape(1, output_size),
            .data = &output_data[i * output_size],
            .device = linear_layer->device
        };
        tensor_eltwise_add(&curr_batch, &linear_layer->bias);
    }
}


static uint32_t linear_layer_backward(
    layer_context_t* context,
    const tensor_t* input,
    const tensor_t* output,
    const tensor_t* prev_gradient,
    tensor_t* out_gradient
)
{
    linear_layer_t* linear_layer = (linear_layer_t*)context;


    const tensor_shape_t* input_shape = tensor_get_shape(input);
    const tensor_shape_t* output_shape = tensor_get_shape(output);
    const tensor_shape_t* weights_shape = tensor_get_shape(&linear_layer->weights);


    const float* input_data = tensor_get_data_const(input);
    const float* prev_gradient_data = tensor_get_data_const(prev_gradient);
    float* gradient_data = tensor_get_data(out_gradient);
    float* weights = tensor_get_data(&linear_layer->weights);
    float* bias = tensor_get_data(&linear_layer->bias);
    float* d_weights = tensor_get_data(&linear_layer->d_weights);
    float* d_bias = tensor_get_data(&linear_layer->d_bias);


    const size_t batch_size = input_shape->dims[TENSOR_BATCH_DIM];
    const size_t per_batch_input_size = tensor_size_from_shape(input_shape) / batch_size;
    const size_t output_channels = output_shape->dims[TENSOR_CHANNEL_DIM];
    const size_t weights_size = tensor_size_from_shape(weights_shape);

    /* Calculate gradient for backprop: gradient = prev_gradient * weights.T */
    matrix_product_t2(prev_gradient_data, weights, gradient_data, batch_size, per_batch_input_size,
        output_channels);

    /* Calculate gradient of weights: d_weights = input.T * prev_gradient */
    matrix_product_t1(input_data, prev_gradient_data, d_weights, per_batch_input_size, output_channels,
        batch_size);
    
    /* Calculate gradient of bias */
    tensor_set_zero(&linear_layer->d_bias);
    for (size_t i = 0; i < batch_size; i++) {
        const tensor_t curr_prev_grad = { /* make-shift tensor view */
            .shape = make_tensor_shape(1, output_channels),
            .data = &prev_gradient_data[i * output_channels],
            .device = linear_layer->device
        };
        tensor_eltwise_add(&linear_layer->d_bias, &curr_prev_grad);
    }
    
    /* d_weights /= batch_size */
    tensor_scale(&linear_layer->d_weights, (1.0f / batch_size));
    tensor_scale(&linear_layer->d_bias, (1.0f / batch_size));
}


static uint32_t linear_layer_calc_output_shape(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
)
{
    linear_layer_create_info_t* linear_create_info = (linear_layer_create_info_t*)create_info;

    /* For now implicitly flatten input. Might be benefical to implement an flatten layer in
        future. */
    out_output_shape->dims[TENSOR_BATCH_DIM] = input_shape->dims[TENSOR_BATCH_DIM];
    out_output_shape->dims[TENSOR_CHANNEL_DIM] = linear_create_info->output_size;
    out_output_shape->dims[TENSOR_HEIGHT_DIM] = 1;
    out_output_shape->dims[TENSOR_WIDTH_DIM] = 1;    

    return 0;
}


const layer_impl_t linear_layer_impl = {
    .init_func = linear_layer_init,
    .get_param_func = linear_layer_get_params,
    .deinit_func = linear_layer_deinit,
    .forward_func = linear_layer_forward,
    .backward_func = linear_layer_backward,
    .calc_output_size = linear_layer_calc_output_shape,
    .layer_context_size = sizeof(linear_layer_t)
};
