#include <malloc.h>
#include <string.h>

#include "core/layer.h"
#include "core/layer_impl.h"

#include "util/dnnl_util.h"

#include "tensor/tensor_impl.h"


struct layer_s {
    const layer_impl_t* impl;
    layer_context_t* context;
    const tensor_shape_t* input_shape;
    tensor_shape_t output_shape;
};


uint32_t layer_create(
    layer_t* layer,
    const layer_impl_t* layer_impl,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    device_t device,
    size_t max_batch_size
)
{
    *layer = malloc(sizeof(**layer));
    if (*layer == NULL) {
        return 1;
    }

    (*layer)->impl = layer_impl;
    (*layer)->impl->get_output_shape(&(*layer)->output_shape, create_info, input_shape);


    (*layer)->context = malloc((*layer)->impl->layer_context_size);
    if ((*layer)->context == NULL) {
        free(*layer);
        return 1;
    }
    memset((*layer)->context, 0, (*layer)->impl->layer_context_size);

    if ((*layer)->impl->init_func((*layer)->context, create_info, input_shape,
                                    &(*layer)->output_shape, device) != 0) {
        free((*layer)->context);
        free(*layer);
        return 1;
    }

    return 0;
}


device_t layer_get_device(layer_t layer)
{
    return device_cpu;
}


const tensor_shape_t* layer_get_output_shape(layer_t layer)
{
    return &layer->output_shape;
}


uint32_t layer_get_param_refs(layer_t layer, layer_param_ref_list_t* out_param_refs)
{
    if (layer->impl->get_param_func == NULL) {
        out_param_refs->param_refs = NULL;
        out_param_refs->num_params = 0;
    } else {
        layer->impl->get_param_func(layer->context, out_param_refs);
    }
    return 0;
}


uint32_t layer_forward(
    layer_t layer,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t** out_output
)
{
    return layer->impl->forward_func(layer->context, forward_kind, input, out_output);
}


uint32_t layer_backward(layer_t layer, const tensor_t* prev_gradient, tensor_t** out_gradient)
{
    return layer->impl->backward_func(layer->context, prev_gradient, out_gradient);
}


uint32_t layer_destroy(layer_t layer)
{
    if (layer != NULL) {
        layer->impl->deinit_func(layer->context);
        free(layer->context);
        free(layer);
    }
    return 0;
}
