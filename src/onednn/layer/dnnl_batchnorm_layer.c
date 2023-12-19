
#include "dnnl.h"

#include "util/dnnl_util.h"
#include "util/dnnl_reorder.h"
#include "util/dnnl_assert.h"
#include "util/ai_math.h"

#include <malloc.h>
#include <stdio.h>

#include "context_impl.h"

#include "tensor_impl.h"
#include "log.h"

#include "core/layer_impl.h"
#include "layer/batchnorm_layer.h"


#define NUM_PARAM_TENSORS   2
#define GAMMA_PARAM_IDX    0
#define BETA_PARAM_IDX      1

#define BN_UPDATE_MOMENTUM 0.9f
#define BN_EPS 1e-8f


typedef struct {
    /* parameters */
    tensor_t gamma;
    tensor_t d_gamma;
    tensor_t beta;
    tensor_t d_beta;
    layer_param_ref_t param_refs[NUM_PARAM_TENSORS];
    

    dnnl_primitive_t fwd_train;
    tensor_t mean;
    tensor_t var;
    tensor_t output;
    tensor_t workspace;
    bool output_mem_initialized;

    dnnl_primitive_t fwd_inference;
    tensor_t running_mean;
    tensor_t running_var;

    
    dnnl_primitive_t bwd;
    dnnl_reorder_t bwd_diff_dst_reorder;
    const tensor_t* input;
    tensor_t gradient;
} batchnorm_layer_t;



static uint32_t batchnorm_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
)
{
    batchnorm_layer_t* layer = context;


    /* Allocate parameter tensors */
    dnnl_memory_desc_t param_md;
    const dnnl_dims_t param_dims = {tensor_shape_get_dim(input_shape, TENSOR_CHANNEL_DIM)};
    dnnl_memory_desc_create_with_tag(&param_md, 1, param_dims, dnnl_f32, dnnl_a);

    tensor_from_desc(&layer->gamma, param_md, DNNL_MEMORY_ALLOCATE);
    tensor_from_desc(&layer->d_gamma, param_md, DNNL_MEMORY_ALLOCATE);
    tensor_from_desc(&layer->beta, param_md, DNNL_MEMORY_ALLOCATE);
    tensor_from_desc(&layer->d_beta, param_md, DNNL_MEMORY_ALLOCATE);
    tensor_from_desc(&layer->mean, param_md, DNNL_MEMORY_ALLOCATE);
    tensor_from_desc(&layer->var, param_md, DNNL_MEMORY_ALLOCATE);
    tensor_from_desc(&layer->running_mean, param_md, DNNL_MEMORY_ALLOCATE);
    tensor_from_desc(&layer->running_var, param_md, DNNL_MEMORY_ALLOCATE);

    dnnl_memory_desc_destroy(param_md);


    /* Register the params */
    layer->param_refs[GAMMA_PARAM_IDX].param = &layer->gamma;
    layer->param_refs[GAMMA_PARAM_IDX].gradient = &layer->d_gamma;
    layer->param_refs[BETA_PARAM_IDX].param = &layer->beta;
    layer->param_refs[BETA_PARAM_IDX].gradient = &layer->d_beta;

    
    /* Initialize parameter tensors */
    tensor_fill(&layer->gamma, 1.0f);
    tensor_fill(&layer->beta, 0.0f);
    tensor_fill(&layer->mean, 0.0f);
    tensor_fill(&layer->var, 1.0f);
    tensor_fill(&layer->running_mean, 0.0f);
    tensor_fill(&layer->running_var, 1.0f);

    return 0;
}


static uint32_t batchnorm_layer_get_params(
    layer_context_t* context,
    layer_param_ref_list_t* out_layer_params
)
{
    batchnorm_layer_t* layer = context;
    
    out_layer_params->num_params = NUM_PARAM_TENSORS;
    out_layer_params->param_refs = layer->param_refs;
    
    return 0;
}


static dnnl_primitive_t batchnorm_create_fwd_primitive(
    const_dnnl_memory_desc_t src_md,
    dnnl_prop_kind_t prop_kind
)
{
    dnnl_status_t status = dnnl_success;
    dnnl_engine_t engine = get_dnnl_engine();

    unsigned int flags = dnnl_use_scale | dnnl_use_shift;
    if (prop_kind == dnnl_forward_inference) {
        flags |= dnnl_use_global_stats;
    }

    dnnl_primitive_desc_t fwd_pd;
    status = dnnl_batch_normalization_forward_primitive_desc_create(&fwd_pd, get_dnnl_engine(),
        prop_kind, src_md, src_md, BN_EPS, flags, NULL);
    if (status != dnnl_success) {
        LOG_ERROR("Creating batchnorm fwd pd failed with code %d\n", status);
        return NULL;
    }

    dnnl_primitive_t primitive;
    status = dnnl_primitive_create(&primitive, fwd_pd);
    if (status != dnnl_success) {
        LOG_ERROR("Creating batchnorm fwd primitive failed with code %d\n", status);
    }
    dnnl_primitive_desc_destroy(fwd_pd);

    return primitive;
}


static dnnl_status_t batchnorm_layer_forward_init(
    batchnorm_layer_t* layer,
    const tensor_t* input,
    dnnl_prop_kind_t prop_kind
)
{
    const_dnnl_memory_desc_t src_md = memory_desc_from_tensor(input);

    dnnl_primitive_t* fwd;
    if (prop_kind == dnnl_forward_training) {
        fwd = &layer->fwd_train;
    } else if (prop_kind == dnnl_forward_inference) {
        fwd = &layer->fwd_inference;
    } else {
        LOG_ERROR("Invalid prop kind\n");
        return 1;
    }

    /* create the primitive */
    *fwd = batchnorm_create_fwd_primitive(src_md, prop_kind);

    if (!layer->output_mem_initialized) {
        /* need allocate the destination memory */
        const_dnnl_memory_desc_t dst_md = dnnlutil_primitive_query_md(*fwd, dnnl_query_dst_md, 0);
        tensor_from_desc(&layer->output, dst_md, DNNL_MEMORY_ALLOCATE);

        layer->output_mem_initialized = true;
    }

    if (prop_kind == dnnl_forward_training) {
        const_dnnl_memory_desc_t workspace_md = dnnlutil_primitive_query_md(*fwd,
        dnnl_query_workspace_md, 0);
        if (tensor_from_desc(&layer->workspace, workspace_md, DNNL_MEMORY_ALLOCATE) != 0) {
            LOG_ERROR("Creating workspace tensor failed with code %d\n", dnnl_out_of_memory);
            return dnnl_out_of_memory;
        }
    }


    return dnnl_success;
}


static uint32_t batchnorm_layer_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t** out_output
)
{
    batchnorm_layer_t* layer = context;
    dnnl_status_t status = dnnl_success;


    if (forward_kind == LAYER_FORWARD_TRAINING) {

        /* initialize forward pass on first call to forward */
        if (layer->fwd_train == NULL) {
            status = batchnorm_layer_forward_init(layer, input, dnnl_forward_training);
            if (status != dnnl_success) {
                return status;
            }
        }

        dnnl_stream_t stream = get_dnnl_stream();
        dnnl_exec_arg_t exec_args[] = {
            { DNNL_ARG_SRC, input->mem },
            { DNNL_ARG_SCALE, layer->gamma.mem },
            { DNNL_ARG_SHIFT, layer->beta.mem },
            { DNNL_ARG_DST, layer->output.mem },
            { DNNL_ARG_MEAN, layer->mean.mem },
            { DNNL_ARG_VARIANCE, layer->var.mem },
        };

        status = dnnl_primitive_execute(layer->fwd_train, stream,
            sizeof(exec_args) / sizeof(*exec_args), exec_args);
        if (status != dnnl_success) {
            LOG_ERROR("primitive execute failed with code %d\n", status);
            return 1;
        }
        status = dnnl_stream_wait(stream);
        
        if (status != dnnl_success) {
            LOG_ERROR("stream_wait failed with code %d\n", status);
            return 1;
        }

        /* Need to update running mean and var for inference */
        const size_t channels = tensor_size_from_shape(tensor_get_shape(&layer->mean));
        float* running_mean_data = tensor_get_data(&layer->running_mean);
        float* running_var_data = tensor_get_data(&layer->running_var);
        const float* mean_data = tensor_get_data(&layer->mean);
        const float* var_data = tensor_get_data(&layer->var);
        VectorScale(running_mean_data, BN_UPDATE_MOMENTUM, channels);
        VectorScale(running_var_data, BN_UPDATE_MOMENTUM, channels);
        VectorScaledAdd(running_mean_data, mean_data, 1.0f - BN_UPDATE_MOMENTUM, channels);
        VectorScaledAdd(running_var_data, var_data, 1.0f - BN_UPDATE_MOMENTUM, channels);

    } else if (forward_kind == LAYER_FORWARD_INFERENCE) {

        /* initialize forward pass on first call to forward */
        if (layer->fwd_inference == NULL) {
            status = batchnorm_layer_forward_init(layer, input,
                dnnl_forward_inference);
            if (status != dnnl_success) {
                return status;
            }
        }

        dnnl_stream_t stream = get_dnnl_stream();
        dnnl_exec_arg_t exec_args[] = {
            { DNNL_ARG_SRC, input->mem },
            { DNNL_ARG_SCALE, layer->gamma.mem },
            { DNNL_ARG_SHIFT, layer->beta.mem },
            { DNNL_ARG_MEAN, layer->running_mean.mem },
            { DNNL_ARG_VARIANCE, layer->running_var.mem },
            { DNNL_ARG_DST, layer->output.mem },
        };
        
        status = dnnl_primitive_execute(layer->fwd_inference, stream,
            sizeof(exec_args) / sizeof(*exec_args), exec_args);
        if (status != dnnl_success) {
            LOG_ERROR("primitive execute failed with code %d\n", status);
            return 1;
        }
        status = dnnl_stream_wait(stream);
        
        if (status != dnnl_success) {
            LOG_ERROR("stream_wait failed with code %d\n", status);
            return 1;
        }

    } else {
        LOG_ERROR("Invalid forward kind\n");
        return 1;
    }

    if (out_output != NULL) {
        *out_output = &layer->output;
    }
    layer->input = input;

    return 0;
}


static dnnl_primitive_t batchnorm_create_bwd_primitive(
    const_dnnl_primitive_desc_t fwd_pd,
    const_dnnl_memory_desc_t diff_dst_md
)
{
    dnnl_status_t status = dnnl_success; 
    dnnl_engine_t engine = get_dnnl_engine();


    const_dnnl_memory_desc_t src_md = dnnl_primitive_desc_query_md(fwd_pd, dnnl_query_src_md, 0);

    unsigned int flags = dnnl_use_global_stats | dnnl_use_shift | dnnl_use_scale;
    float eps;
    dnnl_primitive_desc_query(fwd_pd, dnnl_query_epsilon_f32, 0, &eps);


    dnnl_primitive_desc_t pd;
    status = dnnl_batch_normalization_backward_primitive_desc_create(&pd, engine, dnnl_backward,
        src_md, src_md, src_md, eps, flags, fwd_pd, NULL);
    if (status != dnnl_success) {
        LOG_ERROR("Creating batchnorm backward pd failed with code %d\n", status);
        return NULL;
    }

    dnnl_primitive_t primitive = NULL;
    status = dnnl_primitive_create(&primitive, pd);
    if (status != dnnl_success) {
        LOG_ERROR("Creating batchnorm backward primitive failed with code %d\n", status);
    }


    dnnl_primitive_desc_destroy(pd);

    return primitive;
}


static dnnl_status_t batchnorm_layer_backward_init(
    batchnorm_layer_t* layer,
    const tensor_t* prev_gradient
)
{
    dnnl_status_t status = dnnl_success;


    /* create the primitive */
    const_dnnl_primitive_desc_t fwd_pd;
    dnnl_primitive_get_primitive_desc(layer->fwd_train, &fwd_pd);
    layer->bwd = batchnorm_create_bwd_primitive(fwd_pd, prev_gradient->shape.desc);


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



static uint32_t batchnorm_layer_backward(
    layer_context_t* context,
    const tensor_t* prev_gradient,
    tensor_t** out_gradient
)
{
    batchnorm_layer_t* layer = context;
    dnnl_status_t status = dnnl_success;


    /* initialize backward pass on first call to backward */
    if (layer->bwd == NULL) {
        status = batchnorm_layer_backward_init(layer, prev_gradient);
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
        { DNNL_ARG_SRC, layer->input->mem },
        { DNNL_ARG_MEAN, layer->mean.mem },
        { DNNL_ARG_VARIANCE, layer->var.mem },
        { DNNL_ARG_SCALE, layer->gamma.mem },
        // { DNNL_ARG_SHIFT, layer->beta.mem },
        { DNNL_ARG_DIFF_DST, reordered_prev_gradient->mem },
        { DNNL_ARG_DIFF_SRC, layer->gradient.mem },
        { DNNL_ARG_DIFF_SCALE, layer->d_gamma.mem },
        { DNNL_ARG_DIFF_SHIFT, layer->d_beta.mem },
    };

    status = dnnl_primitive_execute(layer->bwd, stream, sizeof(exec_args) / sizeof(*exec_args),
        exec_args);
    if (status != dnnl_success) {
        LOG_ERROR("Executing batchnorm backward primitive failed with code %d\n", status);
        return 1;
    }

    status = dnnl_stream_wait(stream);
    if (status != dnnl_success) {
        LOG_ERROR("stream_wait failed with code %d\n", status);
        return 1;
    }


    /* onednn will sum over the gradients of a batch. Average over them instead to get more
        representative gradient estimate. */
    const size_t batch_size = tensor_shape_get_dim(tensor_get_shape(layer->input), TENSOR_BATCH_DIM);
    const size_t channels = tensor_shape_get_dim(tensor_get_shape(layer->input), TENSOR_CHANNEL_DIM);

    float* d_gamma_data = tensor_get_data(&layer->d_gamma);
    float* d_beta_data = tensor_get_data(&layer->d_beta);
    VectorScale(d_gamma_data, 1.0f / batch_size, channels);
    VectorScale(d_beta_data, 1.0f / batch_size, channels);


    if (*out_gradient != NULL) {
        *out_gradient = &layer->gradient;
    }

    return 0;
}


static uint32_t batchnorm_layer_deinit(layer_context_t* context)
{
    batchnorm_layer_t* layer = context;

    tensor_destory(&layer->gamma);
    tensor_destory(&layer->d_gamma);
    tensor_destory(&layer->beta);
    tensor_destory(&layer->d_beta);

    tensor_destory(&layer->mean);
    tensor_destory(&layer->var);
    tensor_destory(&layer->running_mean);
    tensor_destory(&layer->running_var);


    if (layer->fwd_train != NULL) {
        dnnl_primitive_destroy(layer->fwd_train);
        if (layer->output_mem_initialized) {
            tensor_destory(&layer->output);
            layer->output_mem_initialized = false;
        }
    }

    if (layer->fwd_inference != NULL) {
        dnnl_primitive_destroy(layer->fwd_inference);
        if (layer->output_mem_initialized) {
            tensor_destory(&layer->output);
            layer->output_mem_initialized = false;
        }
    }

    if (layer->bwd != NULL) {
        dnnl_primitive_destroy(layer->bwd);
        dnnl_reorder_destroy(&layer->bwd_diff_dst_reorder);
        tensor_destory(&layer->gradient);
    }

    return 0;
}


static uint32_t batchnorm_layer_get_output_shape(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
)
{
    /* create a primitive on the fly to check the output shape */
    dnnl_primitive_t fwd = batchnorm_create_fwd_primitive(input_shape->desc, dnnl_forward_training);
    if (fwd == NULL) {
        LOG_ERROR("Failed to create batchnorm fwd primitive\n");
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



const layer_impl_t batchnorm_layer_impl = {
    .init_func = batchnorm_layer_init,
    .get_param_func = batchnorm_layer_get_params,
    .deinit_func = batchnorm_layer_deinit,
    .forward_func = batchnorm_layer_forward,
    .backward_func = batchnorm_layer_backward,
    .get_output_shape = batchnorm_layer_get_output_shape,
    .layer_context_size = sizeof(batchnorm_layer_t),
};
