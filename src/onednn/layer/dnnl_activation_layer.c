

// Forward procedure
// 
// 1. Execute eltwise forward primitive
//

// Backward procedure
//
// 1. Reorder diff dst to recommended format
// 2. Execute eltwise backward primitive
//

// Resources needed in addition to the common ones
//
// 1. Eltwise forward primitive
// 2. Reorder primitive for diff dst if needed
// 3. Intermediate memory for reordered diff dst if needed
// 3. Eltwise backward primitive

#include "dnnl.h"

#include "util/dnnl_util.h"
#include "util/dnnl_reorder.h"
#include "util/dnnl_assert.h"

#include <malloc.h>
#include <stdio.h>

#include "context_impl.h"

#include "tensor_impl.h"
#include "log.h"
#include "core/layer_impl.h"
#include "layer/activation_layer.h"


typedef struct {
    dnnl_primitive_t fwd;
    tensor_t output;
    
    dnnl_primitive_t bwd;
    dnnl_reorder_t bwd_diff_dst_reorder;
    tensor_t gradient;
    
    activation_function_kind_t activation_kind;
} activation_layer_t;


static dnnl_alg_kind_t activation_to_alg_kind(activation_function_kind_t act_kind);


static uint32_t activation_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
)
{
    activation_layer_t* layer = context;
    const activation_layer_create_info_t* activation_create_info = create_info;

    layer->activation_kind = activation_create_info->activation_function;

    return 0;
}


static dnnl_primitive_t act_create_fwd_primitive(
    activation_function_kind_t act_kind,
    const_dnnl_memory_desc_t src_md
)
{
    dnnl_status_t status;

    dnnl_alg_kind_t alg_kind = activation_to_alg_kind(act_kind);
    dnnl_engine_t engine = get_dnnl_engine();

    dnnl_primitive_desc_t fwd_pd;
    status = dnnl_eltwise_forward_primitive_desc_create(&fwd_pd, engine, dnnl_forward_training,
        alg_kind, src_md, src_md, 0.0f, 0.0f, NULL);
    if (status != dnnl_success) {
        LOG_ERROR("Creating eltwise fwd pd failed with code %d\n", status);
        return NULL;
    }

    dnnl_primitive_t primitive;
    status = dnnl_primitive_create(&primitive, fwd_pd);
    if (status != dnnl_success) {
        LOG_ERROR("Creating eltwise fwd primitive failed with code %d\n", status);
    }
    dnnl_primitive_desc_destroy(fwd_pd);
    return primitive;
}


static dnnl_status_t activation_layer_forward_init(activation_layer_t* layer, const tensor_t* input)
{
    const_dnnl_memory_desc_t src_md = memory_desc_from_tensor(input);

    /* create the primitive */
    layer->fwd = act_create_fwd_primitive(layer->activation_kind, src_md);

    /* need to allocate the destination memory */
    const_dnnl_memory_desc_t dst_md = dnnlutil_primitive_query_md(layer->fwd, dnnl_query_dst_md, 0);
    tensor_from_desc(&layer->output, dst_md, DNNL_MEMORY_ALLOCATE);

    return dnnl_success;
}


static uint32_t activation_layer_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t** out_output
)
{
    activation_layer_t* layer = context;
    dnnl_status_t status = dnnl_success;


    /* initialize forward pass on first call to forward */
    if (layer->fwd == NULL) {
        status = activation_layer_forward_init(layer, input);
        if (status != dnnl_success) {
            return status;
        }
    }


    dnnl_stream_t stream = get_dnnl_stream();
    dnnl_exec_arg_t exec_args[2] = {
        { DNNL_ARG_SRC, input->mem },
        { DNNL_ARG_DST, layer->output.mem }
    };

    status = dnnl_primitive_execute(layer->fwd, stream, sizeof(exec_args) / sizeof(*exec_args),
        exec_args);
    if (status != dnnl_success) {
        LOG_ERROR("primitive execute failed with code %d\n", status);
        return 1;
    }

    status = dnnl_stream_wait(stream);
    if (status != dnnl_success) {
        LOG_ERROR("stream_wait failed with code %d\n", status);
        return 1;
    }

    if (out_output != NULL) {
        *out_output = &layer->output;
    }

    return 0;
}


static dnnl_primitive_t act_create_bwd_primitive(const_dnnl_primitive_t fwd_primitive)
{
    dnnl_status_t status = dnnl_success; 
    dnnl_engine_t engine = get_dnnl_engine();


    const_dnnl_primitive_desc_t fwd_pd = nn_dnnl_primitive_get_primitive_desc(fwd_primitive);
    dnnl_alg_kind_t alg_kind;
    dnnl_primitive_desc_query(fwd_pd, dnnl_query_alg_kind, 0, &alg_kind);

    /* Want to keep diff memory formats up to the primitive. Probably will choose the same memory
        format as for the output(dst_md) */
    const_dnnl_memory_desc_t dst_md = dnnl_primitive_desc_query_md(fwd_pd, dnnl_query_dst_md, 0);
    dnnl_memory_desc_t diff_src_dst_md_any = dnnlutil_memory_desc_tag_any(dst_md);


    dnnl_primitive_desc_t pd;
    status = dnnl_eltwise_backward_primitive_desc_create(&pd, engine, alg_kind,
        diff_src_dst_md_any, diff_src_dst_md_any, dst_md, 0.0f, 0.0f, fwd_pd, NULL);
    if (status != dnnl_success) {
        LOG_ERROR("Creating eltwise backward pd failed with code %d\n", status);
        dnnl_memory_desc_destroy(diff_src_dst_md_any);
        return NULL;
    }

    dnnl_primitive_t primitive = NULL;
    status = dnnl_primitive_create(&primitive, pd);
    if (status != dnnl_success) {
        LOG_ERROR("Creating eltwise backward primitive failed with code %d\n", status);
    }

    dnnl_memory_desc_destroy(diff_src_dst_md_any);
    return primitive;
}


static dnnl_status_t activation_layer_backward_init(
    activation_layer_t* layer,
    const tensor_t* prev_gradient
)
{
    dnnl_status_t status = dnnl_success;


    /* create the primitive */
    layer->bwd = act_create_bwd_primitive(layer->fwd);


    /* Need to create potential reorders since format for diff_dst was not specified when creating
        the bwd primitive. */
    const_dnnl_memory_desc_t reorder_src_md = memory_desc_from_tensor(prev_gradient);
    const_dnnl_memory_desc_t reorder_dst_md = dnnlutil_primitive_query_md(layer->bwd,
        dnnl_query_diff_dst_md, 0);

    status = dnnl_reorder_create(&layer->bwd_diff_dst_reorder, reorder_src_md, reorder_dst_md);
    if (status != dnnl_success) {
        LOG_ERROR("Setting up diff dst reorder failed with code %d\n", status);
    }

    /* Need to allocate the gradient(diff src) memory */
    const_dnnl_memory_desc_t diff_src_md = dnnlutil_primitive_query_md(layer->bwd,
        dnnl_query_diff_src_md, 0);
    if (tensor_from_desc(&layer->gradient, diff_src_md, DNNL_MEMORY_ALLOCATE) != 0) {
        status = dnnl_out_of_memory;
        LOG_ERROR("Creating gradient tensor failed with code %d\n", status);
    }

    return status;
}



static uint32_t activation_layer_backward(
    layer_context_t* context,
    const tensor_t* prev_gradient,
    tensor_t** out_gradient
)
{
    activation_layer_t* layer = context;
    dnnl_status_t status = dnnl_success;


    /* initialize backward pass on first call to backward */
    if (layer->bwd == NULL) {
        status = activation_layer_backward_init(layer, prev_gradient);
        if (status != dnnl_success) {
            return status;
        }
    }


    /* Potentially need to reorder prev_gradient */
    const tensor_t* reordered_prev_gradient = NULL;
    status = dnnl_reorder_execute(&layer->bwd_diff_dst_reorder, prev_gradient,
        &reordered_prev_gradient);
    if (status != dnnl_success) {
        LOG_ERROR("Reordering of gradient failed with code %d\n", status);
        return 1;
    }

    dnnl_stream_t stream = get_dnnl_stream();
    dnnl_exec_arg_t exec_args[] = {
        { DNNL_ARG_DST, layer->output.mem },
        { DNNL_ARG_DIFF_DST, reordered_prev_gradient->mem },
        { DNNL_ARG_DIFF_SRC, layer->gradient.mem },
    };

    status = dnnl_primitive_execute(layer->bwd, stream, sizeof(exec_args) / sizeof(*exec_args),
        exec_args);
    if (status != dnnl_success) {
        LOG_ERROR("Executing eltwise backward primitive failed with code %d\n", status);
        return 1;
    }

    status = dnnl_stream_wait(stream);
    if (status != dnnl_success) {
        LOG_ERROR("stream_wait failed with code %d\n", status);
        return 1;
    }

    if (*out_gradient != NULL) {
        *out_gradient = &layer->gradient;
    }

    return 0;
}


static uint32_t activation_layer_deinit(layer_context_t* context)
{
    activation_layer_t* layer = context;

    if (layer->fwd != NULL) {
        dnnl_primitive_destroy(layer->fwd);
        tensor_destory(&layer->output);
    }

    if (layer->bwd != NULL) {
        dnnl_primitive_destroy(layer->bwd);
        dnnl_reorder_destroy(&layer->bwd_diff_dst_reorder);
        tensor_destory(&layer->gradient);
    }

    return 0;
}


static uint32_t activation_layer_get_output_shape(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
)
{
    const activation_layer_create_info_t* act_create_info = create_info;


    /* create a primitive on the fly to check the output shape */
    dnnl_primitive_t fwd = act_create_fwd_primitive(act_create_info->activation_function,
        input_shape->desc);
    if (fwd == NULL) {
        LOG_ERROR("Failed to create activation fwd primitive\n");
        return 1;
    }
    const_dnnl_memory_desc_t dst_md = dnnlutil_primitive_query_md(fwd, dnnl_query_dst_md, 0);


    /* write desc to output */
    dnnl_memory_desc_t dst_md_out;
    dnnl_memory_desc_clone(&dst_md_out, dst_md);
    out_output_shape->desc = dst_md_out;


    /* clean */
    dnnl_primitive_destroy(fwd);

    return 0;
}


static dnnl_alg_kind_t activation_to_alg_kind(activation_function_kind_t act_kind)
{
    dnnl_alg_kind_t alg_kind;

    switch(act_kind) {
        case ACTIVATION_FUNCTION_SIGMOID: alg_kind = dnnl_eltwise_logistic_use_dst_for_bwd; break;
        case ACTIVATION_FUNCTION_TANH: alg_kind = dnnl_eltwise_tanh_use_dst_for_bwd; break;
        case ACTIVATION_FUNCTION_RELU: alg_kind = dnnl_eltwise_relu_use_dst_for_bwd; break;
        default: alg_kind = dnnl_alg_kind_undef; break;
    }

    return alg_kind;
}


const layer_impl_t activation_layer_impl = {
    .init_func = activation_layer_init,
    .get_param_func = NULL,
    .deinit_func = activation_layer_deinit,
    .forward_func = activation_layer_forward,
    .backward_func = activation_layer_backward,
    .get_output_shape = activation_layer_get_output_shape,
    .layer_context_size = sizeof(activation_layer_t),
};

