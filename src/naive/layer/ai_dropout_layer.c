#include <malloc.h>

#include "util/ai_random.h"

#include "ai_layer.h"


typedef struct dropout_layer_t {
    float dropout_rate;
} dropout_layer_t;


static uint32_t dropout_layer_init(void* private_data, const AI_LayerCreateInfo* create_info,
    const tensor_shape_t* input_shape, const tensor_shape_t* output_shape);
static uint32_t dropout_layer_forward(void* private_data, const tensor_t* input,
    tensor_t* out_output);
static uint32_t dropout_layer_backward(void* private_data, const tensor_t* input, const tensor_t* output,
    const tensor_t* prev_gradient, tensor_t* out_gradient);
static uint32_t dropout_layer_calc_output_shape(tensor_shape_t* out_output_shape, const void* create_info,
    const tensor_shape_t* input_shape);


const layer_info_t dropout_layer_info = {
    .init_func = dropout_layer_init,
    .deinit_func = NULL, /* not needed */
    .forward_func = dropout_layer_forward,
    .backward_func = dropout_layer_backward,
    .calc_output_size = dropout_layer_calc_output_shape,
    .info_func = NULL, /* not implemented */
    .layer_private_size = sizeof(dropout_layer_t)
};


uint32_t dropout_layer_calc_output_shape(
    tensor_shape_t* out_output_shape,
    const void* create_info,
    const tensor_shape_t* input_shape
)
{
    /* dropout does not change shape */
    *out_output_shape = *input_shape;

    return 0;
}


static uint32_t dropout_layer_init(
    void* private_data,
    const AI_LayerCreateInfo* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
)
{
    dropout_layer_t* dropout_layer = (dropout_layer_t*)private_data;
    AI_DropoutLayerCreateInfo* dropout_create_info = (AI_DropoutLayerCreateInfo*)private_data;


    dropout_layer->dropout_rate = dropout_create_info->dropout_rate;


    return 0;
}


static uint32_t dropout_layer_forward(
    void* private_data,
    const tensor_t* input,
    tensor_t* out_output
)
{
    dropout_layer_t* dropout_layer = (dropout_layer_t*)private_data;


    const tensor_shape_t* shape = tensor_get_shape(input);
    const size_t size = tensor_size_from_shape(shape);


    const float* input_data = tensor_get_data_const(input);
    float* output_data = tensor_get_data(out_output);


    uint32_t is_training = 1; /* TODO: make it part of the base layer */
    if (is_training != 0) {
        for (size_t i = 0; i < size; i++) {
            float should_drop = (float)(AI_RandomUniform(0.0f, 1.0f) > dropout_layer->dropout_rate);
            output_data[i] = input_data[i] * should_drop;
        }
    } else {
        /* Scale down the activations */
        for (size_t i = 0; i < size; i++) {
            output_data[i] = input_data[i] * (1.0f - dropout_layer->dropout_rate);
        }
    }
}


static uint32_t dropout_layer_backward(
    void* private_data,
    const tensor_t* input,
    const tensor_t* output,
    const tensor_t* prev_gradient,
    tensor_t* out_gradient
)
{
    dropout_layer_t* dropout_layer = (dropout_layer_t*)private_data;


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
