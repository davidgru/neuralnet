
#include "dnnl_base_layer.h"

#include "../util/dnnl_util.h"
#include "../util/dnnl_loss.h"

#include "dnnl_activation_layer.h"
#include "dnnl_input_layer.h"
#include "dnnl_linear_layer.h"
#include "dnnl_convolutional_layer.h"
#include "dnnl_pooling_layer.h"
#include "dnnl_reorder_layer.h"

#include <stdio.h>

typedef uint32_t(*layer_create_fn_t)(dnnl_layer_t** layer, void* create_info);

static layer_create_fn_t get_layer_create_fn(dnnl_layer_kind_t layer_kind)
{
    static layer_create_fn_t t[] = {
        dnnl_input_layer_create,
        dnnl_activation_layer_create,
        dnnl_linear_layer_create,
        dnnl_convolutional_layer_create,
        dnnl_pooling_layer_create,
        0,
        dnnl_reorder_layer_create,
    };
    return t[layer_kind];
}


uint32_t dnnl_layer_create(dnnl_layer_t** layer, dnnl_layer_create_info_t* create_info)
{
    uint32_t status = 0;

    // Create the layer
    layer_create_fn_t create_f = get_layer_create_fn(create_info->layer_kind);
    status = create_f(layer, create_info->layer_create_info);

    return status;
}


uint32_t dnnl_layer_fwd_pass_init(dnnl_layer_t* layer, dnnl_layer_t* prev_layer)
{
    return layer->fwd_pass_init(layer, prev_layer);
}

uint32_t dnnl_layer_bwd_pass_init(dnnl_layer_t* layer, dnnl_layer_t* next_layer)
{
    return layer->bwd_pass_init(layer, next_layer);    
}

uint32_t dnnl_layer_bwd_pass_init_loss(dnnl_layer_t* layer, _dnnl_loss_t* loss)
{
    return layer->bwd_pass_init(layer, ((dnnl_loss_t*)loss)->reorder_layer);
}

uint32_t dnnl_layer_set_input_handle(dnnl_layer_t* layer, float* input)
{
    return dnnl_memory_set_data_handle(layer->src_mem, (void*)input);
}

float* dnnl_layer_get_output_handle(dnnl_layer_t* layer)
{
    return dnnl_memory_get_data_handle(layer->dst_mem);
}

uint32_t dnnl_layer_fwd(dnnl_layer_t* layer)
{
    return layer->fwd(layer);
}

uint32_t dnnl_layer_bwd(dnnl_layer_t* layer)
{
    return layer->bwd(layer);
}

uint32_t dnnl_layer_destroy(dnnl_layer_t* layer)
{
    return layer->destroy(layer);
}
