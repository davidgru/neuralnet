#pragma once


#include <stddef.h>
#include <stdint.h>

#include "tensor.h"


typedef void layer_create_info_t;
typedef void layer_context_t;
typedef struct layer_s* layer_t;


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


/* Backend specific layer api*/
typedef struct layer_impl_s layer_impl_t;



uint32_t layer_create(
    layer_t* layer,
    const layer_impl_t* layer_impl,
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
