#include <malloc.h>


#include "ai_layer.h"


struct layer_impl {
    tensor_shape_t input_shape;
    tensor_t output_mem;
    tensor_t gradient_mem;
    tensor_t output;
    tensor_t gradient;
    const tensor_t* input;
    const layer_info_t* primitive_info;
    void* private_data;
    uint8_t is_training;
};


static const layer_info_t* layer_info_from_kind(AI_LayerKind layer_kind);


uint32_t layer_create(
    layer_t* layer,
    const AI_LayerCreateInfo* create_info,
    const tensor_shape_t* input_shape,
    size_t max_batch_size
)
{
    const layer_info_t* primitive_info = layer_info_from_kind(create_info->type);

    *layer = (layer_t)malloc(sizeof(struct layer_impl));
    if (*layer == NULL) {
        return 1;
    }
    
    (*layer)->input_shape = *input_shape;
    (*layer)->primitive_info = primitive_info;


    /* get the output shape of the primitive. Needed for memory allocation */
    tensor_shape_t output_shape;
    primitive_info->calc_output_size(&output_shape, create_info->create_info, input_shape);


    /* allocate owned resources for the maximum batch size. The input batch size must always be
        lower than the maximum batch size */
    tensor_shape_t max_input_shape = *input_shape;
    tensor_shape_t max_output_shape = output_shape;
    max_input_shape.dims[TENSOR_BATCH_DIM] = max_batch_size;
    max_output_shape.dims[TENSOR_BATCH_DIM] = max_batch_size;

    tensor_allocate(&(*layer)->output_mem, &max_output_shape);
    tensor_allocate(&(*layer)->gradient_mem, &max_input_shape);


    /* initialize layer private data */
    (*layer)->private_data = malloc(primitive_info->layer_private_size);
    if ((*layer)->private_data == NULL) {
        free (*layer);
        return 1;
    }
    primitive_info->init_func((*layer)->private_data, create_info->create_info, input_shape,
        &output_shape);

    return 0;
}


const tensor_shape_t* layer_get_output_shape(layer_t layer)
{
    return tensor_get_shape(&layer->output_mem);
}


uint32_t layer_forward(layer_t layer, const tensor_t* input, tensor_t** out_output)
{
    const tensor_shape_t* input_shape = tensor_get_shape(input);

    /* Construct output tensor with input batch size and embed into output_mem memory pool */
    tensor_shape_t output_shape = *tensor_get_shape(&layer->output_mem);
    output_shape.dims[TENSOR_BATCH_DIM] = input_shape->dims[TENSOR_BATCH_DIM];
    tensor_from_memory(&layer->output, &output_shape, tensor_get_data(&layer->output_mem));

    layer->primitive_info->forward_func(layer->private_data, input, &layer->output);
    
    layer->input = input; /* remember input for backward pass */

    *out_output = &layer->output;
    return 0;
}


uint32_t layer_backward(layer_t layer, const tensor_t* prev_gradient, tensor_t** out_gradient)
{
    const tensor_shape_t* prev_grad_shape = tensor_get_shape(prev_gradient);

    /* Construct gradient tensor with input batch size and embed into gradient_mem memory pool */
    tensor_shape_t gradient_shape = *tensor_get_shape(&layer->gradient_mem);
    gradient_shape.dims[TENSOR_BATCH_DIM] = prev_grad_shape->dims[TENSOR_BATCH_DIM];
    tensor_from_memory(&layer->gradient, &gradient_shape, tensor_get_data(&layer->gradient_mem));

    layer->primitive_info->backward_func(layer->private_data, layer->input, &layer->output,
        prev_gradient, &layer->gradient);

    *out_gradient = &layer->gradient;
    return 0;
}


uint32_t layer_destroy(layer_t layer)
{
    if (layer != NULL) {
        tensor_destory(&layer->output_mem);
        tensor_destory(&layer->gradient_mem);
        free(layer->private_data);
        free(layer);
    }
}


static const layer_info_t* layer_info_from_kind(AI_LayerKind layer_kind)
{
    const layer_info_t* layer_info;

    switch (layer_kind) {
        case AI_ACTIVATION_LAYER:
        {
            layer_info = &activation_layer_info;
            break;
        }
        case AI_CONVOLUTIONAL_LAYER:
        {
            layer_info = &convolutional_layer_info;
            break;
        }
        case AI_DROPOUT_LAYER:
        {
            layer_info = &dropout_layer_info;
            break;
        }
        case AI_LINEAR_LAYER:
        {
            layer_info = &linear_layer_info;
            break;
        }
        case AI_POOLING_LAYER:
        {
            layer_info = &pooling_layer_info;
            break;
        }
        default:
        {
            layer_info = NULL;
            break;
        }
    }

    return layer_info;
}
