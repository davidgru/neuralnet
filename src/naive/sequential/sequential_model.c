#include <malloc.h>
#include <string.h>

#include "sequential_model.h"


typedef struct sequential_model_t {
    layer_t* layers;
    size_t num_layers;
    layer_param_ref_list_t param_refs;
} sequential_model_t;


static uint32_t sequential_model_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
);

static uint32_t sequential_model_get_params(
    layer_context_t* context,
    layer_param_ref_list_t* out_layer_params
);

static uint32_t sequential_model_deinit(layer_context_t* context);

static uint32_t sequential_model_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t* out_output
);

static uint32_t sequential_model_backward(
    layer_context_t* context,
    const tensor_t* input,
    const tensor_t* output,
    const tensor_t* prev_gradient,
    tensor_t* out_gradient
);

static uint32_t sequential_model_calc_output_shape(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
);


const layer_impl_t sequential_model_impl = {
    .init_func = sequential_model_init,
    .get_param_func = sequential_model_get_params,
    .deinit_func = sequential_model_deinit,
    .forward_func = sequential_model_forward,
    .backward_func = sequential_model_backward,
    .calc_output_size = sequential_model_calc_output_shape,
    .layer_context_size = sizeof(sequential_model_t)
};


static uint32_t sequential_model_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
)
{
    sequential_model_t* model = (sequential_model_t*)context;
    const sequential_model_create_info_t* model_create_info
        = (sequential_model_create_info_t*)create_info;

    const model_desc_t* desc = model_create_info->desc;


    model->layers = (layer_t*)calloc(desc->num_layers, sizeof(layer_t));
    if (model->layers == NULL) {
        return 1;
    }
    model->num_layers = desc->num_layers;

    /* initialize the layers */
    const tensor_shape_t* current_output_shape = input_shape;

    for (size_t i = 0; i < desc->num_layers; i++) {
        const layer_impl_t* layer_impl = desc->entries[i].layer_impl;
        const layer_create_info_t* create_info = desc->entries[i].create_info._const;
        layer_create(&model->layers[i], layer_impl, create_info, current_output_shape,
            model_create_info->max_batch_size);
        current_output_shape = layer_get_output_shape(model->layers[i]);
    }


    /* need to provide a list of parameters for the optimizer and for that need to
        combine all params of the sublayers */

    model->param_refs.num_params = 0;
    for (size_t i = 0; i < model->num_layers; i++) {
        layer_param_ref_list_t current_refs;
        layer_get_param_refs(model->layers[i], &current_refs);
        model->param_refs.num_params += current_refs.num_params;
    }

    model->param_refs.param_refs = (layer_param_ref_t*)calloc(
        model->param_refs.num_params, sizeof(layer_param_ref_t));
    if (model->param_refs.param_refs == NULL) {
        return 1;
    }

    size_t current_ref = 0;
    for (size_t i = 0; i < model->num_layers; i++) {
        layer_param_ref_list_t current_refs;
        layer_get_param_refs(model->layers[i], &current_refs);
        memcpy(&model->param_refs.param_refs[current_ref],
            current_refs.param_refs, current_refs.num_params
            * sizeof(layer_param_ref_t));
        current_ref += current_refs.num_params;
    }


    return 0;
}


static uint32_t sequential_model_get_params(
    layer_context_t* context,
    layer_param_ref_list_t* out_layer_params
)
{
    sequential_model_t* model = (sequential_model_t*)context;

    *out_layer_params = model->param_refs;

    return 0;
}


static uint32_t sequential_model_deinit(layer_context_t* context)
{
    sequential_model_t* model = (sequential_model_t*)context;

    for (size_t i = 0; i < model->num_layers; i++) {
        layer_destroy(model->layers[i]);
    }
    free(model->layers);

    if (model->param_refs.param_refs != NULL) {
        free(model->param_refs.param_refs);
    }
}


static uint32_t sequential_model_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t* out_output
)
{
    sequential_model_t* model = (sequential_model_t*)context;

    const tensor_t* current_input = input;
    tensor_t* output = NULL;

    for (size_t i = 0; i < model->num_layers; i++) {
        layer_forward(model->layers[i], forward_kind, current_input, &output);
        current_input = output;
    }
    tensor_copy(out_output, output);
    return 0;
}


static uint32_t sequential_model_backward(
    layer_context_t* context,
    const tensor_t* input,
    const tensor_t* output,
    const tensor_t* prev_gradient,
    tensor_t* out_gradient
)
{
    sequential_model_t* model = (sequential_model_t*)context;

    const tensor_t* gradient = prev_gradient;
    for (int32_t k = model->num_layers - 1; k >= 0; k--) {
        tensor_t* next_gradient;
        layer_backward(model->layers[k], gradient, &next_gradient);
        gradient = next_gradient;
    }
    tensor_copy(out_gradient, gradient);
    return 0;
}


static uint32_t sequential_model_calc_output_shape(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
)
{
    const sequential_model_create_info_t* model_create_info
        = (sequential_model_create_info_t*)create_info;
    const model_desc_t* desc = model_create_info->desc;

    tensor_shape_t current_input_shape = *input_shape;
    for (size_t i = 0; i < desc->num_layers; i++) {
        desc->entries[i].layer_impl->calc_output_size(out_output_shape,
            desc->entries[i].create_info._const, &current_input_shape);
        current_input_shape = *out_output_shape;
    }

    return 0;
}
