#include <malloc.h>

#include "util/ai_random.h"

#include "ai_layer.h"
#include "ai_dropout_layer.h"


typedef struct dropout_layer_t {
    float dropout_rate;
} dropout_layer_t;


static uint32_t dropout_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
);

static uint32_t dropout_layer_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t* out_output
);

static uint32_t dropout_layer_backward(
    layer_context_t* context,
    const tensor_t* input,
    const tensor_t* output,
    const tensor_t* prev_gradient,
    tensor_t* out_gradient
);

static uint32_t dropout_layer_calc_output_shape(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
);


const layer_impl_t dropout_layer_impl = {
    .init_func = dropout_layer_init,
    .get_param_func = NULL, /* no params */
    .deinit_func = NULL, /* not needed */
    .forward_func = dropout_layer_forward,
    .backward_func = dropout_layer_backward,
    .calc_output_size = dropout_layer_calc_output_shape,
    .layer_context_size = sizeof(dropout_layer_t)
};


static uint32_t dropout_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
)
{
    dropout_layer_t* dropout_layer = (dropout_layer_t*)context;
    dropout_layer_create_info_t* dropout_create_info = (dropout_layer_create_info_t*)create_info;


    dropout_layer->dropout_rate = dropout_create_info->dropout_rate;


    return 0;
}


static uint32_t dropout_layer_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t* out_output
)
{
    dropout_layer_t* dropout_layer = (dropout_layer_t*)context;


    const tensor_shape_t* shape = tensor_get_shape(input);
    const size_t size = tensor_size_from_shape(shape);


    const float* input_data = tensor_get_data_const(input);
    float* output_data = tensor_get_data(out_output);


    if (forward_kind == LAYER_FORWARD_INFERENCE) {
        /* Scale down the activations */
        for (size_t i = 0; i < size; i++) {
            output_data[i] = input_data[i] * (1.0f - dropout_layer->dropout_rate);
        }
    } else if (forward_kind == LAYER_FORWARD_TRAINING) {
        /* randomly drop connections */
        for (size_t i = 0; i < size; i++) {
            float should_keep = (float)(AI_RandomUniform(0.0f, 1.0f) > dropout_layer->dropout_rate);
            output_data[i] = input_data[i] * should_keep;
        }
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


    const tensor_shape_t* shape = tensor_get_shape(input);
    const size_t size = tensor_size_from_shape(shape);


    const float* output_data = tensor_get_data_const(output);
    const float* prev_gradient_data = tensor_get_data_const(prev_gradient);
    float* gradient_data = tensor_get_data(out_gradient);


    /* set gradient to zero if value was dropped in forward pass. */
    for (size_t i = 0; i < size; i++) {
        gradient_data[i] = prev_gradient_data[i] * (float)(output_data[i] != 0);
    }
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
