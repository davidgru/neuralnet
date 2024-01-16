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
#include "layer/dropout_layer.h"

#include "random.h"


typedef struct {
    dropout_layer_create_info_t config;

    dnnl_primitive_t fwd_train;
    dnnl_primitive_t fwd_inference;
    dnnl_primitive_t bwd;

    tensor_t mask;
    tensor_t output;
    tensor_t gradient;
    bool output_allocated;

} dropout_layer_t;


static uint32_t dropout_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
)
{
    dropout_layer_t* layer = context;

    layer->config = *(const dropout_layer_create_info_t*)create_info;

    return 0;
}


static dnnl_status_t dropout_layer_forward_train_init(dropout_layer_t* layer, const tensor_t* input)
{
    const_dnnl_memory_desc_t src_md = memory_desc_from_tensor(input);
    const int src_ndims = dnnlutil_memory_desc_get_ndims(src_md);
    const dnnl_dims_t* src_dims = dnnlutil_memory_desc_get_dims(src_md);

    /* allocate mask with same dimensions as src but nchw format */
    dnnl_memory_desc_t mask_md;
    dnnl_memory_desc_create_with_tag(&mask_md, src_ndims, *src_dims, dnnl_f32,
        packed_format_tag_from_ndims(src_ndims));

    tensor_from_desc(&layer->mask, mask_md, DNNL_MEMORY_ALLOCATE);


    dnnl_status_t status;


    /* create binary mul primitive */
    dnnl_primitive_desc_t pd;
    status = dnnl_binary_primitive_desc_create(&pd, get_dnnl_engine(), dnnl_binary_mul, src_md,
        mask_md, src_md, NULL);
    dnnl_memory_desc_destroy(mask_md);
    if (status != dnnl_success) {
        LOG_ERROR("Creating binary pd failed with code %d\n", status);
        return status;
    }

    status = dnnl_primitive_create(&layer->fwd_train, pd);
    dnnl_primitive_desc_destroy(pd);
    if (status != dnnl_success) {
        LOG_ERROR("Creating binary fwd primitive failed with code %d\n", status);
        return status;
    }


    /* need to allocate the destination memory */
    if (!layer->output_allocated) {
        const_dnnl_memory_desc_t dst_md = dnnlutil_primitive_query_md(layer->fwd_train,
            dnnl_query_dst_md, 0);
        tensor_from_desc(&layer->output, dst_md, DNNL_MEMORY_ALLOCATE);
        layer->output_allocated = true;
    }
    

    return dnnl_success;
}


static dnnl_status_t dropout_layer_forward_inf_init(dropout_layer_t* layer, const tensor_t* input)
{
    const_dnnl_memory_desc_t src_md = memory_desc_from_tensor(input);
    const int src_ndims = dnnlutil_memory_desc_get_ndims(src_md);
    const dnnl_dims_t* src_dims = dnnlutil_memory_desc_get_dims(src_md);

    dnnl_status_t status;


    /* create eltiwse linear primitive */
    dnnl_primitive_desc_t pd;
    status = dnnl_eltwise_forward_primitive_desc_create(&pd, get_dnnl_engine(),
        dnnl_forward_training, dnnl_eltwise_linear, src_md, src_md, 1.0f - layer->config.dropout_rate,
        0.0f, NULL);
    if (status != dnnl_success) {
        LOG_ERROR("Creating eltwise linear pd failed with code %d\n", status);
        return status;
    }

    status = dnnl_primitive_create(&layer->fwd_inference, pd);
    dnnl_primitive_desc_destroy(pd);
    if (status != dnnl_success) {
        LOG_ERROR("Creating eltwise linear fwd primitive failed with code %d\n", status);
        return status;
    }


    /* need to allocate the destination memory */
    if (!layer->output_allocated) {
        const_dnnl_memory_desc_t dst_md = dnnlutil_primitive_query_md(layer->fwd_inference,
            dnnl_query_dst_md, 0);
        tensor_from_desc(&layer->output, dst_md, DNNL_MEMORY_ALLOCATE);
        layer->output_allocated = true;
    }


    return dnnl_success;
}


static uint32_t dropout_layer_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t** out_output
)
{
    dropout_layer_t* layer = context;
    dnnl_status_t status = dnnl_success;

    if (forward_kind == LAYER_FORWARD_TRAINING) {

        /* initialize forward pass on first call to forward */
        if (layer->fwd_train == NULL) {
            status = dropout_layer_forward_train_init(layer, input);
            if (status != dnnl_success) {
                return status;
            }
        }

        /* generate dropout mask*/
        random_mask(&layer->mask, 1.0f - layer->config.dropout_rate);

        dnnl_exec_arg_t exec_args[] = {
            { DNNL_ARG_SRC_0, input->mem },
            { DNNL_ARG_SRC_1, layer->mask.mem },
            { DNNL_ARG_DST, layer->output.mem },
        };

        status = dnnl_primitive_execute(layer->fwd_train, get_dnnl_stream(),
            sizeof(exec_args) / sizeof(*exec_args), exec_args);
        if (status != dnnl_success) {
            LOG_ERROR("dropout train primitive execute failed with code %d\n", status);
            return 1;
        }

    } else if (forward_kind == LAYER_FORWARD_INFERENCE) {

        /* initialize forward pass on first call to forward */
        if (layer->fwd_inference == NULL) {
            status = dropout_layer_forward_inf_init(layer, input);
            if (status != dnnl_success) {
                return status;
            }
        }

        dnnl_exec_arg_t exec_args[] = {
            { DNNL_ARG_SRC, input->mem },
            { DNNL_ARG_DST, layer->output.mem },
        };

        status = dnnl_primitive_execute(layer->fwd_inference, get_dnnl_stream(),
            sizeof(exec_args) / sizeof(*exec_args), exec_args);
        if (status != dnnl_success) {
            LOG_ERROR("dropout inference primitive execute failed with code %d\n", status);
            return 1;
        }

    } else {
        LOG_ERROR("Invalid forward kind\n");
        return 1;
    }

    


    status = dnnl_stream_wait(get_dnnl_stream());
    if (status != dnnl_success) {
        LOG_ERROR("stream_wait failed with code %d\n", status);
        return 1;
    }

    if (out_output != NULL) {
        *out_output = &layer->output;
    }

    return 0;
}


static dnnl_status_t dropout_layer_backward_init(
    dropout_layer_t* layer,
    const tensor_t* prev_gradient
)
{
    const_dnnl_memory_desc_t diff_dst_md = memory_desc_from_tensor(prev_gradient);
    const_dnnl_memory_desc_t mask_md = memory_desc_from_tensor(&layer->mask);


    dnnl_status_t status;


    /* create binary mul primitive */
    dnnl_primitive_desc_t pd;
    status = dnnl_binary_primitive_desc_create(&pd, get_dnnl_engine(), dnnl_binary_mul, diff_dst_md,
        mask_md, diff_dst_md, NULL);
    if (status != dnnl_success) {
        LOG_ERROR("Creating binary pd failed with code %d\n", status);
        return status;
    }

    status = dnnl_primitive_create(&layer->bwd, pd);
    dnnl_primitive_desc_destroy(pd);
    if (status != dnnl_success) {
        LOG_ERROR("Creating binary fwd primitive failed with code %d\n", status);
        return status;
    }

    /* need to allocate the gradient(diff_src) memory */
    tensor_from_desc(&layer->gradient, diff_dst_md, DNNL_MEMORY_ALLOCATE);

    return status;
}



static uint32_t dropout_layer_backward(
    layer_context_t* context,
    const tensor_t* prev_gradient,
    tensor_t** out_gradient
)
{
    dropout_layer_t* layer = context;
    dnnl_status_t status = dnnl_success;


    /* initialize backward pass on first call to backward */
    if (layer->bwd == NULL) {
        status = dropout_layer_backward_init(layer, prev_gradient);
        if (status != dnnl_success) {
            return status;
        }
    }


    /* Potentially need to reorder prev_gradient */
    // const tensor_t* reordered_prev_gradient = NULL;
    // status = dnnl_reorder_execute(&layer->bwd_diff_dst_reorder, prev_gradient,
    //     &reordered_prev_gradient);
    // if (status != dnnl_success) {
    //     LOG_ERROR("Reordering of gradient failed with code %d\n", status);
    //     return 1;
    // }

    dnnl_stream_t stream = get_dnnl_stream();
    dnnl_exec_arg_t exec_args[] = {
        { DNNL_ARG_SRC_0, prev_gradient->mem },
        { DNNL_ARG_SRC_1, layer->mask.mem },
        { DNNL_ARG_DST, layer->gradient.mem },
    };

    status = dnnl_primitive_execute(layer->bwd, stream, sizeof(exec_args) / sizeof(*exec_args),
        exec_args);
    if (status != dnnl_success) {
        LOG_ERROR("Executing binary backward primitive failed with code %d\n", status);
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


static uint32_t dropout_layer_deinit(layer_context_t* context)
{
    dropout_layer_t* layer = context;

    if (layer->fwd_train != NULL) {
        dnnl_primitive_destroy(layer->fwd_train);
        tensor_destory(&layer->output);
        layer->output_allocated = false;
        tensor_destory(&layer->mask);
    }

    if (layer->fwd_inference != NULL) {
        dnnl_primitive_destroy(layer->fwd_inference);
        if (layer->output_allocated) {
            tensor_destory(&layer->output);
        }
    }

    if (layer->bwd != NULL) {
        dnnl_primitive_destroy(layer->bwd);
        tensor_destory(&layer->gradient);
    }

    return 0;
}


static uint32_t dropout_layer_get_output_shape(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
)
{
    /* Simply copy since dropout will not affect the shape */
    dnnl_memory_desc_t dst_md;
    dnnl_memory_desc_clone(&dst_md, input_shape->desc);
    out_output_shape->desc = dst_md;

    return 0;
}



const layer_impl_t dropout_layer_impl = {
    .init_func = dropout_layer_init,
    .get_param_func = NULL,
    .deinit_func = dropout_layer_deinit,
    .forward_func = dropout_layer_forward,
    .backward_func = dropout_layer_backward,
    .get_output_shape = dropout_layer_get_output_shape,
    .layer_context_size = sizeof(dropout_layer_t),
};
