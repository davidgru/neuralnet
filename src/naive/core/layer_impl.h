#pragma once 


#include "core/layer.h"


typedef uint32_t (*layer_init_func_t)(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
);


typedef uint32_t (*layer_get_param_func_t)(
    layer_context_t* context,
    layer_param_ref_list_t* out_layer_params
);


typedef uint32_t (*layer_forward_func_t)(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t* out_output
);

typedef uint32_t (*layer_backward_func_t)(
    layer_context_t* context,
    const tensor_t* input,
    const tensor_t* output,
    const tensor_t* prev_gradient,
    tensor_t* out_gradient
);

typedef uint32_t (*layer_deinit_func_t)(layer_context_t* context);

typedef uint32_t (*layer_calc_output_size_func_t)(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
);


struct layer_impl_s {
    layer_init_func_t init_func;
    layer_get_param_func_t get_param_func;
    layer_deinit_func_t deinit_func;
    layer_forward_func_t forward_func;
    layer_backward_func_t backward_func;
    layer_calc_output_size_func_t calc_output_size;
    size_t layer_context_size;
};
