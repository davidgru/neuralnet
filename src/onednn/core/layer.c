#include <malloc.h>

#include "tensor_impl.h"

#include "core/layer.h"


struct layer_s {
    tensor_shape_t input_shape;
    tensor_t output_mem;
    tensor_t gradient_mem;
    tensor_t output;
    tensor_t gradient;
    const tensor_t* input;
    const layer_impl_t* impl;
    layer_context_t* context;
};


uint32_t layer_create(
    layer_t* layer,
    const layer_impl_t* layer_impl,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    size_t max_batch_size
)
{
    *layer = (layer_t)malloc(sizeof(struct layer_s));
    if (*layer == NULL) {
        return 1;
    }
    
    (*layer)->input_shape = *input_shape;
    (*layer)->impl = layer_impl;


    /* get the output shape of the primitive. Needed for memory allocation */
    tensor_shape_t output_shape;
    layer_impl->calc_output_size(&output_shape, create_info, input_shape);


    /* allocate owned resources for the maximum batch size. The input batch size must always be
        lower than the maximum batch size */
    tensor_shape_t max_input_shape = *input_shape;
    tensor_shape_t max_output_shape = output_shape;
    max_input_shape.dims[TENSOR_BATCH_DIM] = max_batch_size;
    max_output_shape.dims[TENSOR_BATCH_DIM] = max_batch_size;

    tensor_allocate(&(*layer)->output_mem, &max_output_shape);
    tensor_allocate(&(*layer)->gradient_mem, &max_input_shape);


    /* initialize layer private data */
    (*layer)->context = (layer_context_t*)malloc(layer_impl->layer_context_size);
    if ((*layer)->context == NULL) {
        free (*layer);
        return 1;
    }
    layer_impl->init_func((*layer)->context, create_info, input_shape,
        &output_shape);

    return 0;
}


const tensor_shape_t* layer_get_output_shape(layer_t layer)
{
    return tensor_get_shape(&layer->output_mem);
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


uint32_t layer_forward(layer_t layer, layer_forward_kind_t forward_kind, const tensor_t* input, tensor_t** out_output)
{
    const tensor_shape_t* input_shape = tensor_get_shape(input);

    /* Construct output tensor with input batch size and embed into output_mem memory pool */
    tensor_shape_t output_shape = *tensor_get_shape(&layer->output_mem);
    output_shape.dims[TENSOR_BATCH_DIM] = input_shape->dims[TENSOR_BATCH_DIM];
    tensor_from_memory(&layer->output, &output_shape, tensor_get_data(&layer->output_mem));

    layer->impl->forward_func(layer->context, forward_kind, input, &layer->output);
    
    layer->input = input; /* remember input for backward pass */

    if (out_output != NULL) {
        *out_output = &layer->output;
    }

    return 0;
}


uint32_t layer_backward(layer_t layer, const tensor_t* prev_gradient, tensor_t** out_gradient)
{
    const tensor_shape_t* prev_grad_shape = tensor_get_shape(prev_gradient);

    /* Construct gradient tensor with input batch size and embed into gradient_mem memory pool */
    tensor_shape_t gradient_shape = *tensor_get_shape(&layer->gradient_mem);
    gradient_shape.dims[TENSOR_BATCH_DIM] = prev_grad_shape->dims[TENSOR_BATCH_DIM];
    tensor_from_memory(&layer->gradient, &gradient_shape, tensor_get_data(&layer->gradient_mem));

    layer->impl->backward_func(layer->context, layer->input, &layer->output,
        prev_gradient, &layer->gradient);

    if (out_gradient != NULL) {
        *out_gradient = &layer->gradient;
    }
    return 0;
}


uint32_t layer_destroy(layer_t layer)
{
    if (layer != NULL) {
        tensor_destory(&layer->output_mem);
        tensor_destory(&layer->gradient_mem);
        free(layer->context);
        free(layer);
    }
}
