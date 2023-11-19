
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
    AI_DropoutLayerCreateInfo* _create_info = (AI_DropoutLayerCreateInfo*)create_info;

    const size_t input_size = prev_layer->output_width * prev_layer->output_height * prev_layer->output_channels;
    const size_t output_size = input_size;

    const size_t size = sizeof(dropout_layer_t) + (input_size + output_size) * prev_layer->mini_batch_size * sizeof(float);
    *layer = (AI_Layer*)malloc(size);

    dropout_layer_t* _layer = (dropout_layer_t*)*layer;

    _layer->hdr.input_width = prev_layer->output_width;
    _layer->hdr.input_height = prev_layer->output_height;
    _layer->hdr.input_channels = prev_layer->output_channels;
    _layer->hdr.output_width = _layer->hdr.input_width;
    _layer->hdr.output_height = _layer->hdr.input_height;
    _layer->hdr.output_channels = _layer->hdr.input_channels;
    _layer->hdr.mini_batch_size = prev_layer->mini_batch_size;
    _layer->hdr.forward = dropout_layer_forward;
    _layer->hdr.backward = dropout_layer_backward;
    _layer->hdr.deinit = dropout_layer_deinit;
    _layer->hdr.is_training = 1;

    _layer->hdr.output = (float*)(_layer + 1);
    _layer->hdr.gradient = _layer->hdr.output + output_size * _layer->hdr.mini_batch_size;
    _layer->hdr.input = 0;
    _layer->hdr.prev_gradient = 0;

    _layer->dropout_rate = _create_info->dropout_rate;

}


static void dropout_layer_forward(AI_Layer* layer)
{
    dropout_layer_t* _layer = (dropout_layer_t*)layer;

    const size_t input_size = _layer->hdr.input_width * _layer->hdr.input_height * _layer->hdr.input_channels * _layer->hdr.mini_batch_size;

    if (_layer->hdr.is_training) {
        // Drop out some activations
        for (size_t i = 0; i < input_size; i++)
            _layer->hdr.output[i] = _layer->hdr.input[i] * (AI_RandomUniform(0.0f, 1.0f) > _layer->dropout_rate);
    }
    else {
        // Scale down the activations
        for (size_t i = 0; i < input_size; i++)
            _layer->hdr.output[i] = _layer->hdr.input[i] * (1.0f - _layer->dropout_rate);
    }
}

static void dropout_layer_backward(AI_Layer* layer)
{
    dropout_layer_t* _layer = (dropout_layer_t*)layer;
    
    const size_t input_size = _layer->hdr.input_width * _layer->hdr.input_height * _layer->hdr.input_channels * _layer->hdr.mini_batch_size;

    for (size_t i = 0; i < input_size; i++)
        _layer->hdr.gradient[i] = _layer->hdr.prev_gradient[i] * (_layer->hdr.output != 0);
}

static void dropout_layer_deinit(AI_Layer* layer)
{
    dropout_layer_t* _layer = (dropout_layer_t*)layer;

    if (_layer)
        free(_layer);
}
