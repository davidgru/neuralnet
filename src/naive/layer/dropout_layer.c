#include <malloc.h>

#include "random.h"

#include "tensor/tensor_impl.h"
#include "tensor/tensor_math.h"
#include "layer/dropout_layer.h"


typedef struct dropout_layer_t {
    tensor_t mask;
    float dropout_rate;
    device_t device;
} dropout_layer_t;


static uint32_t dropout_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape,
    device_t device
)
{
    dropout_layer_t* dropout_layer = (dropout_layer_t*)context;
    dropout_layer_create_info_t* dropout_create_info = (dropout_layer_create_info_t*)create_info;

    dropout_layer->dropout_rate = dropout_create_info->dropout_rate;
    dropout_layer->device = device;
    tensor_allocate_device(&dropout_layer->mask, input_shape, device);

    return 0;
}

static uint32_t dropout_layer_deinit(layer_context_t* context)
{
    dropout_layer_t* dropout_layer = (dropout_layer_t*)context;
    tensor_destory(&dropout_layer->mask);
}


static uint32_t dropout_layer_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t* out_output
)
{
    dropout_layer_t* dropout_layer = (dropout_layer_t*)context;

    if (forward_kind == LAYER_FORWARD_INFERENCE) {
        /* Scale down the activations */
        tensor_set_zero(out_output);
        tensor_scaled_add(out_output, input, 1.0f - dropout_layer->dropout_rate);
    } else if (forward_kind == LAYER_FORWARD_TRAINING) {
        tensor_copy(out_output, input);
        /* generate dropout mask */
        tensor_random_mask(&dropout_layer->mask, 1.0f - dropout_layer->dropout_rate); 
        tensor_eltwise_mul(out_output, &dropout_layer->mask); /* apply mask */
    } else {
        /* Error */
    }
}


static uint32_t dropout_layer_backward(
    layer_context_t* context,
    const tensor_t* input,
    const tensor_t* output,
    const tensor_t* prev_gradient,
    tensor_t* out_gradient
)
{
    dropout_layer_t* dropout_layer = (dropout_layer_t*)context;

    /* apply mask to gradient */
    tensor_copy(out_gradient, prev_gradient);
    tensor_eltwise_mul(out_gradient, &dropout_layer->mask);
}


static uint32_t dropout_layer_calc_output_shape(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
)
{
    /* dropout does not change shape */
    *out_output_shape = *input_shape;

    return 0;
}


const layer_impl_t dropout_layer_impl = {
    .init_func = dropout_layer_init,
    .get_param_func = NULL, /* no params */
    .deinit_func = dropout_layer_deinit,
    .forward_func = dropout_layer_forward,
    .backward_func = dropout_layer_backward,
    .calc_output_size = dropout_layer_calc_output_shape,
    .layer_context_size = sizeof(dropout_layer_t)
};
