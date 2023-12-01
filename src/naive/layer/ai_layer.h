#pragma once


#include <stddef.h>
#include <stdint.h>

#include "tensor.h"

#include "ai_base_layer.h"


struct AI_Layer;


typedef struct layer_param_ref {
    tensor_t* param;
    tensor_t* gradient;
} layer_param_ref_t;


typedef struct layer_param_ref_list {
    layer_param_ref_t* param_refs;
    size_t num_params;
} layer_param_ref_list_t;


typedef enum {
    LAYER_FORWARD_INFERENCE,
    LAYER_FORWARD_TRAINING,
} layer_forward_kind_t;



typedef uint32_t (*layer_init_func_t)(
    void* private_data,
    const AI_LayerCreateInfo* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
);


typedef uint32_t (*layer_get_param_func_t)(
    void* private_data,
    layer_param_ref_list_t* out_layer_params
);


typedef uint32_t (*layer_forward_func_t)(
    void* private_data,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t* out_output
);

typedef uint32_t (*layer_backward_func_t)(
    void* private_data,
    const tensor_t* input,
    const tensor_t* output,
    const tensor_t* prev_gradient,
    tensor_t* out_gradient
);

typedef uint32_t (*layer_info_func_t)(AI_Layer* layer);
typedef uint32_t (*layer_deinit_func_t)(void* private_data);
typedef uint32_t (*layer_calc_output_size_func_t)(
    tensor_shape_t* out_output_shape,
    const void* create_info,
    const tensor_shape_t* input_shape);



typedef struct {
    layer_init_func_t init_func;
    layer_get_param_func_t get_param_func;
    layer_deinit_func_t deinit_func;
    layer_forward_func_t forward_func;
    layer_backward_func_t backward_func;
    layer_calc_output_size_func_t calc_output_size;
    layer_info_func_t info_func;
    size_t layer_private_size;
} layer_info_t;


typedef void layer_create_info_t;


typedef struct layer_s* layer_t;


uint32_t layer_create(
    layer_t* layer,
    const layer_info_t* layer_impl,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    size_t max_batch_size
);


const tensor_shape_t* layer_get_output_shape(layer_t layer);


uint32_t layer_get_param_refs(layer_t layer, layer_param_ref_list_t* out_param_refs);


uint32_t layer_forward(
    layer_t layer,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t** out_output
);


uint32_t layer_backward(layer_t layer, const tensor_t* prev_gradient, tensor_t** out_gradient);


uint32_t layer_destroy(layer_t layer);
