
#include "ai_input_layer.h"

#include "malloc.h"

typedef AI_Layer ai_input_layer_t;

static void input_layer_deinit(AI_Layer* layer);
static void input_layer_forward(AI_Layer* layer);
static void input_layer_backward(AI_Layer* layer);

void input_layer_init(AI_Layer** layer, void* create_info, AI_Layer* prev_layer)
{
    AI_InputLayerCreateInfo* _create_info = (AI_InputLayerCreateInfo*)create_info;

    *layer = (AI_Layer*)malloc(sizeof(ai_input_layer_t));
    ai_input_layer_t* _layer = (ai_input_layer_t*)*layer;

    _layer->input_width = _create_info->input_width;
    _layer->input_height = _create_info->input_height;
    _layer->input_channels = _create_info->input_channels;
    _layer->output_width = _layer->input_width;
    _layer->output_height = _layer->input_height;
    _layer->output_channels = _layer->input_channels;
    _layer->mini_batch_size = _create_info->batch_size;

    _layer->input = 0;
    _layer->output = 0;
    _layer->gradient = 0;
    _layer->prev_gradient = 0;

    _layer->forward = input_layer_forward;
    _layer->backward = input_layer_backward;
    _layer->deinit = input_layer_deinit;
}


static void input_layer_forward(AI_Layer* layer)
{

}

static void input_layer_backward(AI_Layer* layer)
{

}

static void input_layer_deinit(AI_Layer* layer)
{
    ai_input_layer_t* _layer = (ai_input_layer_t*)layer;
    if (_layer)
        free(_layer);
}
