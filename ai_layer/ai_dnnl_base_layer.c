
#include "ai_dnnl_base_layer.h"

#include "../ai_util/ai_dnnl_util.h"
#include "../ai_util/ai_dnnl_loss.h"

#include "ai_dnnl_activation_layer.h"
#include "ai_dnnl_input_layer.h"
#include "ai_dnnl_linear_layer.h"
#include "ai_dnnl_convolutional_layer.h"
#include "ai_dnnl_pooling_layer.h"
#include "ai_dnnl_reorder_layer.h"

#include <stdio.h>

typedef uint32_t(*layer_create_fn_t)(ai_dnnl_layer_t** layer, void* create_info);

static layer_create_fn_t get_layer_create_fn(ai_dnnl_layer_kind_t layer_kind)
{
    static layer_create_fn_t t[] = {
        ai_dnnl_input_layer_create,
        ai_dnnl_activation_layer_create,
        ai_dnnl_linear_layer_create,
        ai_dnnl_convolutional_layer_create,
        ai_dnnl_pooling_layer_create,
        0,
        ai_dnnl_reorder_layer_create,
    };
    return t[layer_kind];
}


uint32_t ai_dnnl_layer_create(ai_dnnl_layer_t** layer, ai_dnnl_layer_create_info_t* create_info)
{
    uint32_t status = 0;

    // Create the layer
    layer_create_fn_t create_f = get_layer_create_fn(create_info->layer_kind);
    status = create_f(layer, create_info->layer_create_info);

    return status;
}


uint32_t ai_dnnl_layer_fwd_pass_init(ai_dnnl_layer_t* layer, ai_dnnl_layer_t* prev_layer)
{
    return layer->fwd_pass_init(layer, prev_layer);
}

uint32_t ai_dnnl_layer_bwd_pass_init(ai_dnnl_layer_t* layer, ai_dnnl_layer_t* next_layer)
{
    return layer->bwd_pass_init(layer, next_layer);    
}

uint32_t ai_dnnl_layer_bwd_pass_init_loss(ai_dnnl_layer_t* layer, _ai_dnnl_loss_t* loss)
{
    return layer->bwd_pass_init(layer, ((ai_dnnl_loss_t*)loss)->reorder_layer);
}

uint32_t ai_dnnl_layer_set_input_handle(ai_dnnl_layer_t* layer, float* input)
{
    return dnnl_memory_set_data_handle(layer->src_mem, (void*)input);
}

float* ai_dnnl_layer_get_output_handle(ai_dnnl_layer_t* layer)
{
    return ai_dnnl_memory_get_data_handle(layer->dst_mem);
}

uint32_t ai_dnnl_layer_fwd(ai_dnnl_layer_t* layer)
{
    return layer->fwd(layer);
}

uint32_t ai_dnnl_layer_bwd(ai_dnnl_layer_t* layer)
{
    return layer->bwd(layer);
}

uint32_t ai_dnnl_layer_destroy(ai_dnnl_layer_t* layer)
{
    return layer->destroy(layer);
}
