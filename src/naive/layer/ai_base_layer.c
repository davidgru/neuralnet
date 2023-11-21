
#include "ai_base_layer.h"

#include "ai_input_layer.h"
#include "ai_activation_layer.h"
#include "ai_linear_layer.h"
#include "ai_convolutional_layer.h"
#include "ai_pooling_layer.h"
#include "ai_dropout_layer.h"

#include "log.h"

void AI_LayerInit(AI_Layer** layer, AI_LayerCreateInfo* create_info, AI_Layer* prev_layer)
{
    LOG_TRACE("Initializing layer of type %d\n", create_info->type);
    switch (create_info->type) {
        case AI_INPUT_LAYER:
            input_layer_init(layer, create_info->create_info, prev_layer);
            break;
        case AI_ACTIVATION_LAYER:
            activation_layer_init(layer, create_info->create_info, prev_layer);
            break;
        case AI_LINEAR_LAYER:
            linear_layer_init(layer, create_info->create_info, prev_layer);
            break;
        case AI_CONVOLUTIONAL_LAYER:
            convolutional_layer_init(layer, create_info->create_info, prev_layer);
            break;
        case AI_POOLING_LAYER:
            pooling_layer_init(layer, create_info->create_info, prev_layer);
            break;
        case AI_DROPOUT_LAYER:
            dropout_layer_init(layer, create_info->create_info, prev_layer);

    }
}

void AI_LayerLink(AI_Layer* layer, AI_Layer* prev_layer, AI_Layer* next_layer)
{
    if (prev_layer)
        layer->input = &prev_layer->output;
    else
        layer->input = 0;

    if (next_layer)
        layer->prev_gradient = &next_layer->gradient;
    else
        layer->prev_gradient = 0;
}


void AI_LayerForward(AI_Layer* layer)
{
    layer->forward(layer);
}

void AI_LayerInfo(AI_Layer* layer)
{
    if (layer->info) {
        layer->info(layer);
    }
}

void AI_LayerBackward(AI_Layer* layer)
{
    layer->backward(layer);
}


void AI_LayerDeinit(AI_Layer* layer)
{
    layer->deinit(layer);
}
