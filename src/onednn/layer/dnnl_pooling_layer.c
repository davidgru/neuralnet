

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
#include "layer/pooling_layer.h"


typedef struct {
    pooling_layer_create_info_t config;

    dnnl_primitive_t fwd;
    tensor_t output;
    tensor_t workspace;
    
    dnnl_primitive_t bwd;
    dnnl_reorder_t bwd_diff_dst_reorder;
    tensor_t gradient;
} pooling_layer_t;


static dnnl_alg_kind_t pooling_to_alg_kind(pooling_kind_t pooling_kind);


static uint32_t pooling_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
)
{
    pooling_layer_t* layer = context;
    layer->config = *(const pooling_layer_create_info_t*)create_info;

    return 0;
}


static dnnl_primitive_t pooling_create_fwd_primitive(
    const_dnnl_memory_desc_t src_md,
    const pooling_layer_create_info_t* config
)
{
    dnnl_status_t status = dnnl_success;


    dnnl_memory_desc_t dst_md_any;
    const dnnl_dims_t dst_dims = {
        dnnlutil_memory_desc_get_dim(src_md, TENSOR_BATCH_DIM),
        dnnlutil_memory_desc_get_dim(src_md, TENSOR_CHANNEL_DIM),
        dnnlutil_memory_desc_get_dim(src_md, TENSOR_HEIGHT_DIM) / config->kernel_width,
        dnnlutil_memory_desc_get_dim(src_md, TENSOR_WIDTH_DIM) / config->kernel_width,
    };
    dnnl_memory_desc_create_with_tag(&dst_md_any, 4, dst_dims,
        dnnl_f32, dnnl_format_tag_any);


    dnnl_alg_kind_t alg_kind = pooling_to_alg_kind(config->pooling_operation);
    dnnl_engine_t engine = get_dnnl_engine();

    const dnnl_dims_t strides = {config->kernel_width, config->kernel_width};
    const dnnl_dims_t kernel = {config->kernel_width, config->kernel_width};
    const dnnl_dims_t dilates = {0, 0}; /* unused */
    const dnnl_dims_t padding_l = {0, 0}; /* unused */
    const dnnl_dims_t padding_r = {0, 0}; /* unused */


    dnnl_primitive_desc_t fwd_pd;
    status = dnnl_pooling_forward_primitive_desc_create(&fwd_pd, engine, dnnl_forward_training,
        alg_kind, src_md, dst_md_any, strides, kernel, dilates, padding_l, padding_r, NULL);
    if (status != dnnl_success) {
        LOG_ERROR("Creating pooling fwd pd failed with code %d\n", status);
        return NULL;
    }

    dnnl_primitive_t primitive;
    status = dnnl_primitive_create(&primitive, fwd_pd);
    if (status != dnnl_success) {
        LOG_ERROR("Creating pooling fwd primitive failed with code %d\n", status);
    }
    dnnl_primitive_desc_destroy(fwd_pd);
    return primitive;
}


static dnnl_status_t pooling_layer_forward_init(pooling_layer_t* layer, const tensor_t* input)
{
    const_dnnl_memory_desc_t src_md = memory_desc_from_tensor(input);

    /* create the primitive */
    layer->fwd = pooling_create_fwd_primitive(src_md, &layer->config);

    /* need allocate the destination memory */
    const_dnnl_memory_desc_t dst_md = dnnlutil_primitive_query_md(layer->fwd, dnnl_query_dst_md, 0);
    tensor_from_desc(&layer->output, dst_md, DNNL_MEMORY_ALLOCATE);

    /* need allocate workspace memory */
    const_dnnl_memory_desc_t workspace_md = dnnlutil_primitive_query_md(layer->fwd,
        dnnl_query_workspace_md, 0);
    tensor_from_desc(&layer->workspace, workspace_md, DNNL_MEMORY_ALLOCATE);

    return dnnl_success;
}


static uint32_t pooling_layer_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t** out_output
)
{
    pooling_layer_t* layer = context;
    dnnl_status_t status = dnnl_success;


    /* initialize forward pass on first call to forward */
    if (layer->fwd == NULL) {
        status = pooling_layer_forward_init(layer, input);
        if (status != dnnl_success) {
            return status;
        }
    }


    dnnl_stream_t stream = get_dnnl_stream();
    dnnl_exec_arg_t exec_args[] = {
        { DNNL_ARG_SRC, input->mem },
        { DNNL_ARG_DST, layer->output.mem },
        { DNNL_ARG_WORKSPACE, layer->workspace.mem }
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


static dnnl_primitive_t pooling_create_bwd_primitive(
    const_dnnl_primitive_desc_t fwd_pd,
    const_dnnl_memory_desc_t diff_dst_md
)
{
    dnnl_status_t status = dnnl_success; 
    dnnl_engine_t engine = get_dnnl_engine();


    /* Want to keep diff memory formats up to the primitive. Probably will choose the same memory
        format as for the output(dst_md) */
    const_dnnl_memory_desc_t src_md = dnnl_primitive_desc_query_md(fwd_pd, dnnl_query_src_md, 0);
    dnnl_memory_desc_t diff_src_md_any = dnnlutil_memory_desc_tag_any(src_md);


    dnnl_alg_kind_t alg_kind;
    const dnnl_dims_t* strides;
    const dnnl_dims_t* kernel;
    const dnnl_dims_t* dilates;
    const dnnl_dims_t* padding_l;
    const dnnl_dims_t* padding_r;

    dnnl_primitive_desc_query(fwd_pd, dnnl_query_alg_kind, 0, &alg_kind);
    dnnl_primitive_desc_query(fwd_pd, dnnl_query_strides, 0, &strides);
    dnnl_primitive_desc_query(fwd_pd, dnnl_query_kernel, 0, &kernel);
    dnnl_primitive_desc_query(fwd_pd, dnnl_query_dilations, 0, &dilates);
    dnnl_primitive_desc_query(fwd_pd, dnnl_query_padding_l, 0, &padding_l);
    dnnl_primitive_desc_query(fwd_pd, dnnl_query_padding_r, 0, &padding_r);

    dnnl_primitive_desc_t pd;
    status = dnnl_pooling_backward_primitive_desc_create(&pd, engine, alg_kind, diff_src_md_any,
        diff_dst_md, *strides, *kernel, *dilates, *padding_l, *padding_r, fwd_pd, NULL);
    dnnl_memory_desc_destroy(diff_src_md_any);
    if (status != dnnl_success) {
        LOG_ERROR("Creating pooling backward pd failed with code %d\n", status);
        return NULL;
    }

    dnnl_primitive_t primitive = NULL;
    status = dnnl_primitive_create(&primitive, pd);
    if (status != dnnl_success) {
        LOG_ERROR("Creating pooling backward primitive failed with code %d\n", status);
    }


    dnnl_primitive_desc_destroy(pd);

    return primitive;
}


static dnnl_status_t pooling_layer_backward_init(
    pooling_layer_t* layer,
    const tensor_t* prev_gradient
)
{
    dnnl_status_t status = dnnl_success;


    /* create the primitive */
    const_dnnl_primitive_desc_t fwd_pd;
    dnnl_primitive_get_primitive_desc(layer->fwd, &fwd_pd);
    layer->bwd = pooling_create_bwd_primitive(fwd_pd, prev_gradient->shape.desc);


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



static uint32_t pooling_layer_backward(
    layer_context_t* context,
    const tensor_t* prev_gradient,
    tensor_t** out_gradient
)
{
    pooling_layer_t* layer = context;
    dnnl_status_t status = dnnl_success;


    /* initialize backward pass on first call to backward */
    if (layer->bwd == NULL) {
        status = pooling_layer_backward_init(layer, prev_gradient);
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
        { DNNL_ARG_DIFF_SRC, layer->gradient.mem },
        { DNNL_ARG_DIFF_DST, reordered_prev_gradient->mem },
        { DNNL_ARG_WORKSPACE, layer->workspace.mem }
    };

    status = dnnl_primitive_execute(layer->bwd, stream, sizeof(exec_args) / sizeof(*exec_args),
        exec_args);
    if (status != dnnl_success) {
        LOG_ERROR("Executing pooling backward primitive failed with code %d\n", status);
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


static uint32_t pooling_layer_deinit(layer_context_t* context)
{
    pooling_layer_t* layer = context;

    if (layer->fwd != NULL) {
        dnnl_primitive_destroy(layer->fwd);
        tensor_destory(&layer->output);
        tensor_destory(&layer->workspace);
    }

    if (layer->bwd != NULL) {
        dnnl_primitive_destroy(layer->bwd);
        dnnl_reorder_destroy(&layer->bwd_diff_dst_reorder);
        tensor_destory(&layer->gradient);
    }

    return 0;
}


static uint32_t pooling_layer_get_output_shape(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
)
{
    /* create a primitive on the fly to check the output shape */
    dnnl_primitive_t fwd = pooling_create_fwd_primitive(input_shape->desc, create_info);
    if (fwd == NULL) {
        LOG_ERROR("Failed to create pooling fwd primitive\n");
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


static dnnl_alg_kind_t pooling_to_alg_kind(pooling_kind_t pooling_kind)
{
    dnnl_alg_kind_t alg_kind;

    switch(pooling_kind) {
        case POOLING_AVERAGE: alg_kind = dnnl_pooling_avg_include_padding; break;
        case POOLING_MAX: alg_kind = dnnl_pooling_max; break;
        default: alg_kind = dnnl_alg_kind_undef; break;
    }

    return alg_kind;
}


const layer_impl_t pooling_layer_impl = {
    .init_func = pooling_layer_init,
    .get_param_func = NULL,
    .deinit_func = pooling_layer_deinit,
    .forward_func = pooling_layer_forward,
    .backward_func = pooling_layer_backward,
    .get_output_shape = pooling_layer_get_output_shape,
    .layer_context_size = sizeof(pooling_layer_t),
};
