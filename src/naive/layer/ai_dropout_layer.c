
#include "ai_dropout_layer.h"

#include "util/ai_random.h"

#include <malloc.h>

typedef struct dropout_layer_t {
    AI_Layer hdr;
    float dropout_rate;
} dropout_layer_t;

static void dropout_layer_forward(AI_Layer* layer);
static void dropout_layer_backward(AI_Layer* layer);
static void dropout_layer_deinit(AI_Layer* layer);


uint32_t dropout_layer_init(AI_Layer** layer, void* create_info, AI_Layer* prev_layer)
{
    AI_DropoutLayerCreateInfo* dropout_create_info = (AI_DropoutLayerCreateInfo*)create_info;

    *layer = (AI_Layer*)malloc(sizeof(dropout_layer_t));
    if (*layer == NULL) {
        return 1;
    }

    dropout_layer_t* dropout_layer = (dropout_layer_t*)*layer;

    /* fill header information */
    dropout_layer->hdr.input_shape = prev_layer->output_shape;
    dropout_layer->hdr.output_shape = prev_layer->output_shape; /* shape does not change*/

    /* allocate owned memory */
    tensor_allocate(&dropout_layer->hdr.output, &dropout_layer->hdr.output_shape);
    tensor_allocate(&dropout_layer->hdr.gradient, &dropout_layer->hdr.input_shape);

    /* virtual functions */
    dropout_layer->hdr.forward = dropout_layer_forward;
    dropout_layer->hdr.backward = dropout_layer_backward;
    dropout_layer->hdr.info = NULL;
    dropout_layer->hdr.deinit = dropout_layer_deinit;

    dropout_layer->dropout_rate = dropout_create_info->dropout_rate;

    return 0;
}


static void dropout_layer_forward(AI_Layer* layer)
{
    dropout_layer_t* dropout_layer = (dropout_layer_t*)layer;


    const tensor_shape_t* shape = tensor_get_shape(dropout_layer->hdr.input);
    const size_t size = tensor_size_from_shape(shape);


    float* input = tensor_get_data(dropout_layer->hdr.input);
    float* output = tensor_get_data(&dropout_layer->hdr.output);


    if (dropout_layer->hdr.is_training != 0) {
        for (size_t i = 0; i < size; i++) {
            float should_drop = (float)(AI_RandomUniform(0.0f, 1.0f) > dropout_layer->dropout_rate);
            output[i] = input[i] * should_drop;
        }
    } else {
        /* Scale down the activations */
        for (size_t i = 0; i < size; i++) {
            output[i] = input[i] * (1.0f - dropout_layer->dropout_rate);
        }
    }
}


static void dropout_layer_backward(AI_Layer* layer)
{
    dropout_layer_t* dropout_layer = (dropout_layer_t*)layer;


    const tensor_shape_t* shape = tensor_get_shape(dropout_layer->hdr.input);
    const size_t size = tensor_size_from_shape(shape);


    float* output = tensor_get_data(&dropout_layer->hdr.output);
    float* gradient = tensor_get_data(&dropout_layer->hdr.gradient);
    float* prev_gradient = tensor_get_data(dropout_layer->hdr.prev_gradient);


    /* set gradient to zero if value was dropped in forward pass. */
    for (size_t i = 0; i < size; i++) {
        gradient[i] = prev_gradient[i] * (float)(output[i] != 0);
    }
}


static void dropout_layer_deinit(AI_Layer* layer)
{
    dropout_layer_t* dropout_layer = (dropout_layer_t*)layer;
    if (dropout_layer != NULL) {
        tensor_destory(&dropout_layer->hdr.output);
        tensor_destory(&dropout_layer->hdr.gradient);
    }
}
