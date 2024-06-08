

// fwd proc
//
// 1. Reorder src if necessary
// 2. Reorder weights if necessary
// 3. Execute inner product fwd primitive

// bwd data proc
//
// 1. Reorder diff dst if necessary
// 2. Reorder weights if necessary
// 3. Execute inner product bwd data primitive
//
// bwd weights proc
//
// 1. Reorder diff dst if necessary
// 2. Reorder weights if necessary
// 3. Execute inner product bwd weights primitive
// 4. Reorder diff weights if necessary
// 5. Update weights and bias

#include <string.h>

#include "context_impl.h"
#include "log.h"

#include "util/ai_math.h"
#include "util/dnnl_reorder.h"
#include "util/dnnl_util.h"

#include "layer/convolutional_layer.h"
#include "core/layer_impl.h"


#define NUM_PARAM_TENSORS   2
#define WEIGHT_PARAM_IDX    0
#define BIAS_PARAM_IDX      1

#define conv_output_size(input_size, kernel_size, stride, dilation, padding) \
    (((input_size) + 2 * (padding) - (dilation) * ((kernel_size) - 1) - 1) / (stride) + 1)


typedef struct {
    convolutional_layer_create_info_t config;

    /* remember input for gradient calculation */
    const tensor_t* input;

    /* parameters */
    tensor_t weight;
    tensor_t d_weight;
    tensor_t bias;
    tensor_t d_bias;
    layer_param_ref_t param_refs[NUM_PARAM_TENSORS];

    /* forward */
    dnnl_primitive_t fwd;
    dnnl_reorder_t fwd_reorder_src;
    dnnl_reorder_t fwd_reorder_weight;
    tensor_t output;
    tensor_t workspace;

    /* bwd data */
    dnnl_primitive_t bwd_data;
    dnnl_reorder_t bwd_data_reorder_weight;
    dnnl_reorder_t bwd_data_reorder_diff_dst;
    tensor_t gradient;

    /* bwd weights */
    dnnl_primitive_t bwd_weights;
    dnnl_reorder_t bwd_weights_reorder_src;
    dnnl_reorder_t bwd_weights_reorder_diff_dst;
    /* for potential reorder in case weights_md != bwd_weights_diff_weights_md */
    tensor_t bwd_weights_diff_weight;
    dnnl_reorder_t bwd_weights_reorder_diff_weight;
} conv_layer_t;



static dnnl_memory_desc_t weight_md_from_src_md(
    const_dnnl_memory_desc_t src_md,
    const convolutional_layer_create_info_t* config
);


static uint32_t conv_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape,
    device_t device
)
{
    conv_layer_t* layer = context;
    layer->config = *(const convolutional_layer_create_info_t*)create_info;


    /* Allocate parameter tensors */    
    dnnl_memory_desc_t weight_md = weight_md_from_src_md(input_shape->desc, &layer->config);

    dnnl_memory_desc_t bias_md;
    const dnnl_dims_t bias_dims = {layer->config.output_channels};
    dnnl_memory_desc_create_with_tag(&bias_md, 1, bias_dims,
        dnnl_f32, dnnl_a);

    tensor_from_desc(&layer->weight, weight_md, DNNL_MEMORY_ALLOCATE);
    tensor_from_desc(&layer->d_weight, weight_md, DNNL_MEMORY_ALLOCATE);
    tensor_from_desc(&layer->bias, bias_md, DNNL_MEMORY_ALLOCATE);
    tensor_from_desc(&layer->d_bias, bias_md, DNNL_MEMORY_ALLOCATE);

    dnnl_memory_desc_destroy(weight_md);
    dnnl_memory_desc_destroy(bias_md);

    /* Register parameters for optimizer */
    layer->param_refs[WEIGHT_PARAM_IDX].param = &layer->weight;
    layer->param_refs[WEIGHT_PARAM_IDX].gradient = &layer->d_weight;
    layer->param_refs[BIAS_PARAM_IDX].param = &layer->bias;
    layer->param_refs[BIAS_PARAM_IDX].gradient = &layer->d_bias;


    /* Initialize parameter tensors */
    layer->config.weight_init(&layer->weight);
    layer->config.bias_init(&layer->bias);
    
    return 0;
}


static uint32_t conv_layer_get_params(
    layer_context_t* context,
    layer_param_ref_list_t* out_layer_params
)
{
    conv_layer_t* layer = context;

    out_layer_params->num_params = NUM_PARAM_TENSORS;
    out_layer_params->param_refs = layer->param_refs;

    return 0;
}


/* Allowing reorder of all involved tensors */
static dnnl_primitive_t conv_layer_create_fwd_primitive(
    const tensor_shape_t* input_shape,
    const convolutional_layer_create_info_t* config
)
{
    dnnl_memory_desc_t fwd_src_md_any = dnnlutil_memory_desc_tag_any(input_shape->desc);
    dnnl_memory_desc_t weight_md = weight_md_from_src_md(fwd_src_md_any, config);
    dnnl_memory_desc_t fwd_weight_md_any = dnnlutil_memory_desc_tag_any(weight_md);
    
    dnnl_memory_desc_t fwd_bias_md;
    const dnnl_dims_t bias_dims = { config->output_channels };
    dnnl_memory_desc_create_with_tag(&fwd_bias_md, 1, bias_dims, dnnl_f32, dnnl_a);

    dnnl_memory_desc_t fwd_dst_md_any;
    const dnnl_dims_t output_dims = { 
        dnnlutil_memory_desc_get_dim(fwd_src_md_any, 0),
        config->output_channels,
        conv_output_size(tensor_shape_get_dim(input_shape, TENSOR_HEIGHT_DIM),
            config->filter_height, config->stride_y, 1, config->padding_y),
        conv_output_size(tensor_shape_get_dim(input_shape, TENSOR_WIDTH_DIM),
            config->filter_width, config->stride_x, 1, config->padding_x),
    };
    dnnl_memory_desc_create_with_tag(&fwd_dst_md_any, 4,
        output_dims, dnnl_f32, dnnl_format_tag_any);


    dnnl_status_t status = dnnl_success;
    dnnl_engine_t engine = get_dnnl_engine();

    const dnnl_dims_t strides = {config->stride_y, config->stride_x};
    const dnnl_dims_t dilates = { 0, 0 }; /* not used */
    const dnnl_dims_t padding = {config->padding_y, config->padding_x};

    dnnl_primitive_desc_t pd;
    status  = dnnl_convolution_forward_primitive_desc_create(&pd, engine, dnnl_forward_training,
        dnnl_convolution_auto, fwd_src_md_any, fwd_weight_md_any, fwd_bias_md, fwd_dst_md_any,
        strides, dilates, padding, padding, NULL);

    dnnl_memory_desc_destroy(fwd_src_md_any);
    dnnl_memory_desc_destroy(weight_md);
    dnnl_memory_desc_destroy(fwd_weight_md_any);
    dnnl_memory_desc_destroy(fwd_bias_md);
    dnnl_memory_desc_destroy(fwd_dst_md_any);

    if (status != dnnl_success) {
        LOG_ERROR("Creating conv forward pd failed with code %d\n", status);
        return NULL;
    }

    dnnl_primitive_t primitive;
    status = dnnl_primitive_create(&primitive, pd);
    if (status != dnnl_success) {
        LOG_ERROR("Creating conv forward primitive failed with code %d\n", status);
    }


    dnnl_primitive_desc_destroy(pd);

    return primitive;    
}


static dnnl_status_t conv_layer_forward_init(conv_layer_t* layer, const tensor_t* input)
{
    dnnl_status_t status = dnnl_success;


    /* create the primitive */
    layer->fwd = conv_layer_create_fwd_primitive(tensor_get_shape(input), &layer->config);


    /* need allocate output memory */
    const_dnnl_memory_desc_t dst_md = dnnlutil_primitive_query_md(layer->fwd, dnnl_query_dst_md, 0);
    if (tensor_from_desc(&layer->output, dst_md, DNNL_MEMORY_ALLOCATE) != 0) {
        status = dnnl_out_of_memory;
        LOG_ERROR("Creating output tensor failed with code %d\n", status);
        return status;
    }


    /* need allocate workspace memory */
    const_dnnl_memory_desc_t workspace_md = dnnlutil_primitive_query_md(layer->fwd,
        dnnl_query_workspace_md, 0);
    if (tensor_from_desc(&layer->workspace, workspace_md, DNNL_MEMORY_ALLOCATE) != 0) {
        status = dnnl_out_of_memory;
        LOG_ERROR("Creating workspace tensor failed with code %d\n", status);
        return status;
    }


    /* potential reorder of input */
    const_dnnl_memory_desc_t src_md = memory_desc_from_tensor(input);
    const_dnnl_memory_desc_t fwd_src_md = dnnlutil_primitive_query_md(layer->fwd, dnnl_query_src_md,
        0);
    status = dnnl_reorder_create(&layer->fwd_reorder_src, src_md, fwd_src_md);
    if (status != dnnl_success) {
        LOG_ERROR("Setting up input reorder failed with code %d\n", status);
        return status;
    }


    /* potential reorder of weight */
    const_dnnl_memory_desc_t weight_md = memory_desc_from_tensor(&layer->weight);
    const_dnnl_memory_desc_t fwd_weight_md = dnnlutil_primitive_query_md(layer->fwd,
        dnnl_query_weights_md, 0);
    status = dnnl_reorder_create(&layer->fwd_reorder_weight, weight_md, fwd_weight_md);
    if (status != dnnl_success) {
        LOG_ERROR("Setting up weight reorder failed with code %d\n", status);
        return status;
    }


    return status;
}


static uint32_t conv_layer_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t** out_output
)
{
    conv_layer_t* layer = context;
    dnnl_status_t status = dnnl_success;


    /* initialize forward pass on first call to forward */
    if (layer->fwd == NULL) {
        status = conv_layer_forward_init(layer, input);
        if (status != dnnl_success) {
            return 1;
        }
    }


    /* reorder src */
    const tensor_t* input_reordered;
    status = dnnl_reorder_execute(&layer->fwd_reorder_src, input, &input_reordered);
    if (status != dnnl_success) {
        LOG_ERROR("Reordering of input failed with code %d\n", status);
        return 1;
    }

    /* reorder weight */
    const tensor_t* weight_reordered;
    status = dnnl_reorder_execute(&layer->fwd_reorder_weight, &layer->weight, &weight_reordered);
    if (status != dnnl_success) {
        LOG_ERROR("Reordering of weight failed with code %d\n", status);
        return 1;
    }


    /* execute forward primitive */
    dnnl_stream_t stream = get_dnnl_stream();
    dnnl_exec_arg_t exec_args[] = {
        { DNNL_ARG_SRC, input_reordered->mem },
        { DNNL_ARG_WEIGHTS, weight_reordered->mem },
        { DNNL_ARG_BIAS, layer->bias.mem },
        { DNNL_ARG_DST, layer->output.mem },
    };

    status = dnnl_primitive_execute(layer->fwd, stream, sizeof(exec_args) / sizeof(*exec_args),
        exec_args);
    if (status != dnnl_success) {
        LOG_ERROR("conv primitive execute failed with code %d\n", status);
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
    layer->input = input;

    return 0;    
}


static dnnl_primitive_t conv_layer_create_bwd_data_primitive(
    const_dnnl_primitive_desc_t fwd_pd
)
{
    const_dnnl_memory_desc_t src_md = dnnl_primitive_desc_query_md(fwd_pd, dnnl_query_src_md, 0);
    const_dnnl_memory_desc_t weight_md = dnnl_primitive_desc_query_md(fwd_pd, dnnl_query_weights_md,
        0);
    const_dnnl_memory_desc_t dst_md = dnnl_primitive_desc_query_md(fwd_pd, dnnl_query_dst_md, 0);

    dnnl_memory_desc_t diff_src_md_any = dnnlutil_memory_desc_tag_any(src_md);
    dnnl_memory_desc_t weight_md_any = dnnlutil_memory_desc_tag_any(weight_md);
    dnnl_memory_desc_t diff_dst_md_any = dnnlutil_memory_desc_tag_any(dst_md);


    dnnl_status_t status = dnnl_success;

    const dnnl_dims_t* strides;
    const dnnl_dims_t* dilates;
    const dnnl_dims_t* padding_l;
    const dnnl_dims_t* padding_r;

    dnnl_primitive_desc_query(fwd_pd, dnnl_query_strides, 0, &strides);
    dnnl_primitive_desc_query(fwd_pd, dnnl_query_dilations, 0, &dilates);
    dnnl_primitive_desc_query(fwd_pd, dnnl_query_padding_l, 0, &padding_l);
    dnnl_primitive_desc_query(fwd_pd, dnnl_query_padding_r, 0, &padding_r);


    dnnl_primitive_desc_t pd;
    status = dnnl_convolution_backward_data_primitive_desc_create(&pd, get_dnnl_engine(),
        dnnl_convolution_auto, diff_src_md_any, weight_md_any, diff_dst_md_any, *strides, *dilates,
        *padding_l, *padding_r, fwd_pd, NULL);

    dnnl_memory_desc_destroy(diff_src_md_any);
    dnnl_memory_desc_destroy(weight_md_any);
    dnnl_memory_desc_destroy(diff_dst_md_any);

    if (status != dnnl_success) {
        LOG_ERROR("Creating conv bwd data pd failed with code %d\n", status);
        return NULL;
    }

    dnnl_primitive_t primitive = NULL;
    status = dnnl_primitive_create(&primitive, pd);
    if (status != dnnl_success) {
        LOG_ERROR("Creating conv bwd data primitive failed with code %d\n", status);
    }

    dnnl_primitive_desc_destroy(pd);

    return primitive;
}


static dnnl_status_t conv_layer_backward_data_init(
    conv_layer_t* layer,
    const tensor_t* prev_gradient
)
{
    dnnl_status_t status = dnnl_success;


    /* create the bwd data primitive */
    const_dnnl_primitive_desc_t fwd_pd;
    dnnl_primitive_get_primitive_desc(layer->fwd, &fwd_pd);
    layer->bwd_data = conv_layer_create_bwd_data_primitive(fwd_pd);


    /* need allocate gradient tensor */
    const_dnnl_memory_desc_t bwd_data_diff_src_md = dnnlutil_primitive_query_md(layer->bwd_data,
        dnnl_query_diff_src_md, 0);
    if (tensor_from_desc(&layer->gradient, bwd_data_diff_src_md, DNNL_MEMORY_ALLOCATE) != 0) {
        status = dnnl_out_of_memory;
        LOG_ERROR("Creating gradient tensor failed with code %d\n", status);
        return status;
    }


    /* potential reorder of weight */
    const_dnnl_memory_desc_t weight_md = memory_desc_from_tensor(&layer->weight);
    const_dnnl_memory_desc_t bwd_data_weight_md = dnnlutil_primitive_query_md(layer->bwd_data,
        dnnl_query_weights_md, 0);
    status = dnnl_reorder_create(&layer->bwd_data_reorder_weight, weight_md, bwd_data_weight_md);
    if (status != dnnl_success) {
        LOG_ERROR("Setting up weight reorder failed with code %d\n", status);
        return status;
    }

    /* potential reorder of prev_gradient(diff_dst) */
    const_dnnl_memory_desc_t diff_dst_md = memory_desc_from_tensor(prev_gradient);
    const_dnnl_memory_desc_t bwd_data_diff_dst_md = dnnlutil_primitive_query_md(layer->bwd_data,
        dnnl_query_diff_dst_md, 0);
    status = dnnl_reorder_create(&layer->bwd_data_reorder_diff_dst, diff_dst_md,
        bwd_data_diff_dst_md);
    if (status != dnnl_success) {
        LOG_ERROR("Setting up prev_gradient reorder failed with code %d\n", status);
        return status;
    }

    return status;
}


static dnnl_primitive_t conv_layer_create_bwd_weights_primitive(
    const_dnnl_primitive_desc_t fwd_pd
)
{
    const_dnnl_memory_desc_t src_md = dnnl_primitive_desc_query_md(fwd_pd, dnnl_query_src_md, 0);
    const_dnnl_memory_desc_t weight_md = dnnl_primitive_desc_query_md(fwd_pd, dnnl_query_weights_md,
        0);
    const_dnnl_memory_desc_t bias_md = dnnl_primitive_desc_query_md(fwd_pd, dnnl_query_weights_md,
        1);
    const_dnnl_memory_desc_t dst_md = dnnl_primitive_desc_query_md(fwd_pd, dnnl_query_dst_md, 0);

    dnnl_memory_desc_t src_md_any = dnnlutil_memory_desc_tag_any(src_md);
    dnnl_memory_desc_t diff_weight_md_any = dnnlutil_memory_desc_tag_any(weight_md);
    dnnl_memory_desc_t diff_dst_md_any = dnnlutil_memory_desc_tag_any(dst_md);


    dnnl_status_t status = dnnl_success;

    const dnnl_dims_t* strides;
    const dnnl_dims_t* dilates;
    const dnnl_dims_t* padding_l;
    const dnnl_dims_t* padding_r;

    dnnl_primitive_desc_query(fwd_pd, dnnl_query_strides, 0, &strides);
    dnnl_primitive_desc_query(fwd_pd, dnnl_query_dilations, 0, &dilates);
    dnnl_primitive_desc_query(fwd_pd, dnnl_query_padding_l, 0, &padding_l);
    dnnl_primitive_desc_query(fwd_pd, dnnl_query_padding_r, 0, &padding_r);


    dnnl_primitive_desc_t pd;
    status = dnnl_convolution_backward_weights_primitive_desc_create(&pd, get_dnnl_engine(),
        dnnl_convolution_auto, src_md_any, diff_weight_md_any, bias_md, diff_dst_md_any, *strides,
        *dilates, *padding_l, *padding_r, fwd_pd, NULL);

    dnnl_memory_desc_destroy(src_md_any);
    dnnl_memory_desc_destroy(diff_weight_md_any);
    dnnl_memory_desc_destroy(diff_dst_md_any);

    if (status != dnnl_success) {
        LOG_ERROR("Creating conv bwd weights pd failed with code %d\n", status);
        return NULL;
    }

    dnnl_primitive_t primitive = NULL;
    status = dnnl_primitive_create(&primitive, pd);
    if (status != dnnl_success) {
        LOG_ERROR("Creating conv bwd weights primitive failed with code %d\n", status);
    }

    dnnl_primitive_desc_destroy(pd);

    return primitive;
}


static dnnl_status_t conv_layer_backward_weights_init(
    conv_layer_t* layer,
    const tensor_t* input,
    const tensor_t* prev_gradient
)
{
    dnnl_status_t status = dnnl_success;


    /* create the bwd data primitive */
    const_dnnl_primitive_desc_t fwd_pd;
    dnnl_primitive_get_primitive_desc(layer->fwd, &fwd_pd);
    layer->bwd_weights = conv_layer_create_bwd_weights_primitive(fwd_pd);


    /* potential reorder of input */
    const_dnnl_memory_desc_t src_md = memory_desc_from_tensor(input);
    const_dnnl_memory_desc_t bwd_weights_src_md = dnnlutil_primitive_query_md(layer->bwd_weights,
        dnnl_query_src_md, 0);
    status = dnnl_reorder_create(&layer->bwd_weights_reorder_src, src_md, bwd_weights_src_md);
    if (status != dnnl_success) {
        LOG_ERROR("Setting up input reorder failed with code %d\n", status);
        return status;
    }

    /* potential reorder of prev_gradient */
    const_dnnl_memory_desc_t diff_dst_md = memory_desc_from_tensor(prev_gradient);
    const_dnnl_memory_desc_t bwd_weights_diff_dst_md = dnnlutil_primitive_query_md(
        layer->bwd_weights, dnnl_query_diff_dst_md, 0);
    status = dnnl_reorder_create(&layer->bwd_weights_reorder_diff_dst, diff_dst_md,
        bwd_weights_diff_dst_md);
    if (status != dnnl_success) {
        LOG_ERROR("Setting up prev_gradient reorder failed with code %d\n", status);
        return status;
    }    

    /* potential reorder of diff_weights */
    const_dnnl_memory_desc_t diff_weights_md = memory_desc_from_tensor(&layer->d_weight);
    const_dnnl_memory_desc_t bwd_weights_diff_weight_md = dnnlutil_primitive_query_md(
        layer->bwd_weights, dnnl_query_diff_weights_md, 0);
    if (!dnnl_memory_desc_equal(diff_weights_md, bwd_weights_diff_weight_md)) {

        /* need intermediate diff_weights memory */
        if (tensor_from_desc(&layer->bwd_weights_diff_weight, bwd_weights_diff_weight_md,
            DNNL_MEMORY_ALLOCATE) != 0) {
            status = dnnl_out_of_memory;
            LOG_ERROR("Creating intermediate d_weights memory failed with code %d\n", status);
            return status;
        }

        /* set up reorder to d_weights */
        status = dnnl_reorder_create(&layer->bwd_weights_reorder_diff_weight,
            bwd_weights_diff_weight_md, diff_weights_md);
        if (status != dnnl_success) {
            LOG_ERROR("Setting up d_weight reorder failed with code %d\n", status);
            return status;
        }
        /* (ugly) hack to reorder into d_weights directly */
        layer->bwd_weights_reorder_diff_weight.output = layer->d_weight;
    }

    return status;
}


static dnnl_status_t conv_layer_backward_data(
    conv_layer_t* layer,
    const tensor_t* prev_gradient,
    tensor_t** out_gradient
)
{
    dnnl_status_t status = dnnl_success;


    const tensor_t* weight_reordered;
    status = dnnl_reorder_execute(&layer->bwd_data_reorder_weight, &layer->weight,
        &weight_reordered);
    if (status != dnnl_success) {
        LOG_ERROR("Reordering of weight failed with code %d\n", status);
        return status;
    }

    const tensor_t* prev_gradient_reordered;
    status = dnnl_reorder_execute(&layer->bwd_data_reorder_diff_dst, prev_gradient,
        &prev_gradient_reordered);
    if (status != dnnl_success) {
        LOG_ERROR("Reordering of prev_gradient failed with code %d\n", status);
        return status;
    }


    dnnl_stream_t stream = get_dnnl_stream();
    dnnl_exec_arg_t exec_args[] = {
        { DNNL_ARG_DIFF_SRC, layer->gradient.mem },
        { DNNL_ARG_WEIGHTS, weight_reordered->mem },
        { DNNL_ARG_DIFF_DST, prev_gradient_reordered->mem },
    };

    status = dnnl_primitive_execute(layer->bwd_data, stream, sizeof(exec_args) / sizeof(*exec_args),
        exec_args);
    if (status != dnnl_success) {
        LOG_ERROR("bwd data primitive execute failed with code %d\n", status);
    }

    status = dnnl_stream_wait(stream);
    if (status != dnnl_success) {
        LOG_ERROR("stream_wait failed with code %d\n", status);
        return 1;
    }

    if (out_gradient != NULL) {
        *out_gradient = &layer->gradient;
    }
    
    return status;
}


static dnnl_status_t conv_layer_backward_weights(
    conv_layer_t* layer,
    const tensor_t* input,
    const tensor_t* prev_gradient
)
{
    dnnl_status_t status = dnnl_success;


    const tensor_t* input_reordered;
    status = dnnl_reorder_execute(&layer->bwd_weights_reorder_src, input,
        &input_reordered);
    if (status != dnnl_success) {
        LOG_ERROR("Reordering of input failed with code %d\n", status);
        return status;
    }

    const tensor_t* prev_gradient_reordered;
    status = dnnl_reorder_execute(&layer->bwd_weights_reorder_diff_dst, prev_gradient,
        &prev_gradient_reordered);
    if (status != dnnl_success) {
        LOG_ERROR("Reordering of prev_gradient failed with code %d\n", status);
        return status;
    }

    bool need_diff_weights_reorder = layer->bwd_weights_reorder_diff_weight.primitive != NULL;


    dnnl_stream_t stream = get_dnnl_stream();
    dnnl_exec_arg_t exec_args[] = {
        { DNNL_ARG_SRC, input_reordered->mem },
        { DNNL_ARG_DIFF_WEIGHTS,
            need_diff_weights_reorder ? layer->bwd_weights_diff_weight.mem : layer->d_weight.mem },
        { DNNL_ARG_DIFF_BIAS, layer->d_bias.mem },
        { DNNL_ARG_DIFF_DST, prev_gradient_reordered->mem },
    };

    status = dnnl_primitive_execute(layer->bwd_weights, stream,
        sizeof(exec_args) / sizeof(*exec_args), exec_args);
    if (status != dnnl_success) {
        LOG_ERROR("bwd weights primitive execute failed with code %d\n", status);
    }

    status = dnnl_stream_wait(stream);
    if (status != dnnl_success) {
        LOG_ERROR("stream_wait failed with code %d\n", status);
        return 1;
    }

    if (need_diff_weights_reorder) {
        /* TODO: This is really ugly and needs to be changed. */
        status = dnnl_reorder_execute(&layer->bwd_weights_reorder_diff_weight,
            &layer->bwd_weights_diff_weight, NULL);
        if (status != dnnl_success) {
            LOG_ERROR("Reordering of d_weight failed with code %d\n", status);
            return status;
        }
    }


    /* onednn will sum over the gradients of a batch. Average over them instead to get more
        representative gradient estimate. */
    const tensor_shape_t* input_shape = tensor_get_shape(input);
    const size_t batch_size = tensor_shape_get_dim(input_shape, TENSOR_BATCH_DIM);

    float* d_weight_data = tensor_get_data(&layer->d_weight);
    const size_t weight_size = tensor_size_from_shape(tensor_get_shape(&layer->d_weight));
    VectorScale(d_weight_data, 1.0f / batch_size, weight_size);

    float* d_bias_data = tensor_get_data(&layer->d_bias);
    const size_t bias_size = tensor_size_from_shape(tensor_get_shape(&layer->d_bias));
    VectorScale(d_bias_data, 1.0f / batch_size, bias_size);


    return status;
}


static uint32_t conv_layer_backward(
    layer_context_t* context,
    const tensor_t* prev_gradient,
    tensor_t** out_gradient
)
{
    conv_layer_t* layer = context;
    dnnl_status_t status = dnnl_success;


    /* initialize backward pass on first call to backward */
    if (layer->bwd_data == NULL) {

        status = conv_layer_backward_data_init(layer, prev_gradient);
        if (status != dnnl_success) {
            return 1;
        }

        status = conv_layer_backward_weights_init(layer, layer->input, prev_gradient);
        if (status != dnnl_success) {
            return 1;
        }
    }


    status = conv_layer_backward_data(layer, prev_gradient, out_gradient);
    if (status != dnnl_success) {
        LOG_ERROR("conv layer backward data failed with code %d\n", status);
        return 1;
    }

    status = conv_layer_backward_weights(layer, layer->input, prev_gradient);
    if (status != dnnl_success) {
        LOG_ERROR("conv layer backward weights failed with code %d\n", status);
        return 1;
    }


    status = dnnl_stream_wait(get_dnnl_stream());
    if (status != dnnl_success) {
        LOG_ERROR("stream_wait failed with code %d\n", status);
        return 1;
    }

    return 0;
}


static uint32_t conv_layer_deinit(layer_context_t* context)
{
    conv_layer_t* layer = context;

    tensor_destory(&layer->weight);
    tensor_destory(&layer->d_weight);
    tensor_destory(&layer->bias);
    tensor_destory(&layer->d_bias);

    if (layer->fwd != NULL) {
        dnnl_primitive_destroy(layer->fwd);
        dnnl_reorder_destroy(&layer->fwd_reorder_src);
        dnnl_reorder_destroy(&layer->fwd_reorder_weight);
        tensor_destory(&layer->output);
        tensor_destory(&layer->workspace);
    }

    if (layer->bwd_data != NULL) {
        dnnl_primitive_destroy(layer->bwd_data);
        dnnl_reorder_destroy(&layer->bwd_data_reorder_weight);
        dnnl_reorder_destroy(&layer->bwd_data_reorder_diff_dst);
        tensor_destory(&layer->gradient);
    }

    if (layer->bwd_weights != NULL) {
        dnnl_primitive_destroy(layer->bwd_weights);
        dnnl_reorder_destroy(&layer->bwd_weights_reorder_src);
        dnnl_reorder_destroy(&layer->bwd_weights_reorder_diff_dst);
        if (layer->bwd_weights_reorder_diff_weight.primitive != NULL) {
            dnnl_primitive_destroy(layer->bwd_weights_reorder_diff_weight.primitive);
            tensor_destory(&layer->bwd_weights_diff_weight);
        }
    }

    return 0;
}


static uint32_t conv_layer_get_output_shape(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
)
{
    const convolutional_layer_create_info_t* conv_create_info = create_info;


    /* create a primitive on the fly to check the output shape */
    dnnl_primitive_t fwd = conv_layer_create_fwd_primitive(input_shape,
        conv_create_info);
    if (fwd == NULL) {
        LOG_ERROR("Creating conv forward primitive failed\n");
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


static dnnl_memory_desc_t weight_md_from_src_md(
    const_dnnl_memory_desc_t src_md,
    const convolutional_layer_create_info_t* config
)
{
    const dnnl_dims_t* src_dims = dnnlutil_memory_desc_get_dims(src_md);

    const dnnl_dims_t weight_dims = {
        config->output_channels,
        (*src_dims)[TENSOR_CHANNEL_DIM],
        config->filter_height,
        config->filter_width
    };
    dnnl_memory_desc_t weight_md;
    dnnl_status_t status = dnnl_memory_desc_create_with_tag(&weight_md,
        4, weight_dims, dnnl_f32, dnnl_oihw);
    if (status != dnnl_success) {
        LOG_ERROR("Creating weight_md failed with code %d\n", status);
        return NULL;
    }

    return weight_md;
}



const convolutional_layer_create_info_t conv_default_config = {
    .output_channels = 0,
    .filter_height = 0,
    .filter_width = 0,
    .stride_y = 1,
    .stride_x = 1,
    .padding_y = 0,
    .padding_x = 0,
    .weight_init = winit_xavier,
    .bias_init = winit_zeros,
};


const layer_impl_t convolutional_layer_impl = {
    .init_func = conv_layer_init,
    .get_param_func = conv_layer_get_params,
    .deinit_func = conv_layer_deinit,
    .forward_func = conv_layer_forward,
    .backward_func = conv_layer_backward,
    .get_output_shape = conv_layer_get_output_shape,
    .layer_context_size = sizeof(conv_layer_t),
};
