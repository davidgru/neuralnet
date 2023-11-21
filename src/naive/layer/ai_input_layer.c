
#include "ai_input_layer.h"

#include "malloc.h"


typedef AI_Layer ai_input_layer_t;


static void dummy_forward_backward_info(AI_Layer* layer);
static void input_layer_deinit(AI_Layer* layer);


void input_layer_init(AI_Layer** layer, void* create_info, AI_Layer* prev_layer)
{
    AI_InputLayerCreateInfo* input_create_info = (AI_InputLayerCreateInfo*)create_info;

    *layer = (AI_Layer*)malloc(sizeof(ai_input_layer_t));
    ai_input_layer_t* input_layer = (ai_input_layer_t*)*layer;

    input_layer->input_shape = input_create_info->input_shape;
    input_layer->output_shape = input_create_info->input_shape;

    input_layer->input = NULL;
    input_layer->prev_gradient = NULL;

    input_layer->forward = dummy_forward_backward_info;
    input_layer->backward = dummy_forward_backward_info;
    input_layer->info = dummy_forward_backward_info;
    input_layer->deinit = input_layer_deinit;
}


static void dummy_forward_backward_info(AI_Layer* layer)
{
    /* Do nothing. */
    (void)layer;
}


static void input_layer_deinit(AI_Layer* layer)
{
    ai_input_layer_t* input_layer = (ai_input_layer_t*)layer;
    if (input_layer != NULL) {
        free(input_layer);
    }
}
