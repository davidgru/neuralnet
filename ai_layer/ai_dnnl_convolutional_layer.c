
#include "ai_dnnl_convolutional_layer.h"


#include "../ai_util/ai_random.h"

#include "../ai_util/ai_dnnl_util.h"
#include "../ai_util/ai_dnnl_reorder.h"
#include "../ai_util/ai_dnnl_assert.h"

#include <malloc.h>
#include <math.h>

typedef struct ai_dnnl_convolutional_layer_t {
    ai_dnnl_layer_t hdr;

    size_t KH;
    size_t KW;
    size_t SH;
    size_t SW;
    size_t PT;
    size_t PL;
    size_t PB;
    size_t PR;

    float learning_rate;
    int32_t dummy;

    ai_dnnl_convolutional_layer_weight_init_kind_t weight_init;
    ai_dnnl_convolutional_layer_bias_init_kind_t bias_init;

    // common memory
    dnnl_memory_t weights_mem;
    dnnl_memory_t bias_mem;
    dnnl_memory_t diff_weights_mem;
    dnnl_memory_t diff_bias_mem;
    dnnl_memory_t workspace_mem;

    // fwd
    dnnl_primitive_t fwd;
    dnnl_memory_t fwd_src_mem;
    dnnl_memory_t fwd_weights_mem;

    // bwd data
    dnnl_primitive_t bwd_data;
    dnnl_memory_t bwd_data_diff_dst_mem;
    dnnl_memory_t bwd_data_weights_mem;

    // bwd weights
    dnnl_primitive_t bwd_weights;
    dnnl_memory_t bwd_weights_src_mem;
    dnnl_memory_t bwd_weights_diff_dst_mem;
    dnnl_memory_t bwd_weights_diff_weights_mem;
    
    // reorders
    ai_dnnl_reorder_t fwd_reorder_src;
    ai_dnnl_reorder_t fwd_reorder_weights;
    ai_dnnl_reorder_t bwd_data_reorder_diff_dst;
    ai_dnnl_reorder_t bwd_data_reorder_weights;
    ai_dnnl_reorder_t bwd_weights_reorder_src;
    ai_dnnl_reorder_t bwd_weights_reorder_diff_dst;
    ai_dnnl_reorder_t bwd_weights_reorder_diff_weights;


} ai_dnnl_convolutional_layer_t;


static uint32_t conv_layer_fwd_pass_init(ai_dnnl_layer_t* layer, ai_dnnl_layer_t* prev_layer);
static uint32_t conv_layer_bwd_pass_init(ai_dnnl_layer_t* layer, ai_dnnl_layer_t* next_layer);
static uint32_t conv_layer_fwd(ai_dnnl_layer_t* layer);
static uint32_t conv_layer_bwd(ai_dnnl_layer_t* layer);
static uint32_t conv_layer_destroy(ai_dnnl_layer_t* layer);


typedef float(*conv_weight_init_fn)(size_t KH, size_t KW, size_t IC);
typedef float(*conv_bias_init_fn)(size_t KH, size_t KW, size_t IC);

static float conv_weight_init_xavier(size_t KH, size_t KW, size_t IC);
static float conv_weight_init_he(size_t KH, size_t KW, size_t IC);
static float conv_bias_init_zeros(size_t KH, size_t KW, size_t IC);


// Utility function to get a weight init function given an enum value
static conv_weight_init_fn get_weight_init_function(ai_dnnl_convolutional_layer_weight_init_kind_t weight_init)
{
    static conv_weight_init_fn t[] = {
        conv_weight_init_xavier,
        conv_weight_init_he
    };
    return t[weight_init];
}

// Utility function to get a bias init function given an enum value
static conv_bias_init_fn get_bias_init_function(ai_dnnl_convolutional_layer_bias_init_kind_t bias_init)
{
    static conv_bias_init_fn t[] = {
        conv_bias_init_zeros
    };
    return t[bias_init];
}


uint32_t ai_dnnl_convolutional_layer_create(ai_dnnl_layer_t** layer, void* create_info)
{
    // Allocate memory for the layer
    *layer = (ai_dnnl_layer_t*)malloc(sizeof(ai_dnnl_convolutional_layer_t));

    ai_dnnl_convolutional_layer_t* l = (ai_dnnl_convolutional_layer_t*)*layer;
    ai_dnnl_convolutional_layer_create_info_t* i = (ai_dnnl_convolutional_layer_create_info_t*)create_info;

    // Set hyperparameters of the layer
    l->hdr.OC = i->OC;
    l->KH = i->KH;
    l->KW = i->KW;
    l->SH = i->SH;
    l->SW = i->SW;
    l->PT = i->PT;
    l->PL = i->PL;
    l->PB = i->PB;
    l->PR = i->PR;

    l->learning_rate = i->learning_rate;
    l->weight_init = i->weight_init;
    l->bias_init = i->bias_init;

    l->hdr.fwd_pass_init = conv_layer_fwd_pass_init;
    l->hdr.bwd_pass_init = conv_layer_bwd_pass_init;
    l->hdr.fwd = conv_layer_fwd;
    l->hdr.bwd = conv_layer_bwd;
    l->hdr.destroy = conv_layer_destroy;

    return 0;
}


uint32_t conv_layer_fwd_pass_init(ai_dnnl_layer_t* layer, ai_dnnl_layer_t* prev_layer)
{
    ai_dnnl_convolutional_layer_t* l = (ai_dnnl_convolutional_layer_t*)layer;

    // Set input and output tensor dimensions
    l->hdr.N = prev_layer->N;
    l->hdr.IC = prev_layer->OC;
    l->hdr.IH = prev_layer->OH;
    l->hdr.IW = prev_layer->OW;
    l->hdr.OH = (l->hdr.IH - l->KH + l->PT + l->PB) / l->SH + 1;
    l->hdr.OW = (l->hdr.IW - l->KW + l->PL + l->PR) / l->SW + 1;

    l->hdr.src_mem = prev_layer->dst_mem;
    
    l->hdr.engine = prev_layer->engine;
    l->hdr.stream = prev_layer->stream;

    // 1. Create a convolution fwd primitive

    // 1.1 Create memory descs for all inputs and outputs
    const dnnl_memory_desc_t* src_md = ai_dnnl_memory_get_memory_desc(l->hdr.src_mem);
    
    dnnl_dims_t weights_dims = { l->hdr.OC, l->hdr.IC, l->KH, l->KW };
    dnnl_dims_t bias_dims = { l->hdr.OC };
    dnnl_dims_t dst_dims = { l->hdr.N, l->hdr.OC, l->hdr.OH, l->hdr.OW };
    
    dnnl_dims_t strides = { l->SH, l->SW };
    dnnl_dims_t padding_l = { l->PT, l->PL };
    dnnl_dims_t padding_r = { l->PB, l->PR };

    dnnl_memory_desc_t fwd_src_md_any;
    dnnl_memory_desc_t fwd_weights_md_any;
    dnnl_memory_desc_t bias_md;
    dnnl_memory_desc_t fwd_dst_md_any;
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&fwd_src_md_any, src_md->ndims, src_md->dims, dnnl_f32, dnnl_format_tag_any));
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&fwd_weights_md_any, 4, weights_dims, dnnl_f32, dnnl_format_tag_any));
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&bias_md, 1, bias_dims, dnnl_f32, dnnl_a));
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&fwd_dst_md_any, 4, dst_dims, dnnl_f32, dnnl_format_tag_any));
    
    // 1.2 Create a convolution fwd primitive
    dnnl_convolution_desc_t fwd_d;
    dnnl_primitive_desc_t fwd_pd;
    CHECK_DNNL(dnnl_convolution_forward_desc_init(&fwd_d, dnnl_forward_training, dnnl_convolution_auto, &fwd_src_md_any, &fwd_weights_md_any, &bias_md, &fwd_dst_md_any, strides, padding_l, padding_r));
    CHECK_DNNL(dnnl_primitive_desc_create(&fwd_pd, &fwd_d, 0, l->hdr.engine, 0));
    CHECK_DNNL(dnnl_primitive_create(&l->fwd, fwd_pd));

    // 2. Create weights and bias memory
    dnnl_memory_desc_t weights_md;
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&weights_md, 4, weights_dims, dnnl_f32, dnnl_oihw));
    CHECK_DNNL(dnnl_memory_create(&l->weights_mem, &weights_md, l->hdr.engine, DNNL_MEMORY_ALLOCATE));
    CHECK_DNNL(dnnl_memory_create(&l->bias_mem, &bias_md, l->hdr.engine, DNNL_MEMORY_ALLOCATE));
    // Init weights
    conv_weight_init_fn weight_init = get_weight_init_function(l->weight_init);
    float* weights = ai_dnnl_memory_get_data_handle(l->weights_mem);
    for (size_t i = 0; i < l->hdr.OC * l->hdr.IC * l->KH * l->KW; i++)
        weights[i] = weight_init(l->KH, l->KW, l->hdr.IC);
    // Init bias
    conv_bias_init_fn bias_init = get_bias_init_function(l->bias_init);
    float* bias = ai_dnnl_memory_get_data_handle(l->bias_mem);
    for (size_t i = 0; i < l->hdr.OC; i++)
        bias[i] = bias_init(l->KH, l->KW, l->hdr.IC);

    // 3. Set up reorder between src and fwd_src memory
    const dnnl_memory_desc_t* fwd_src_md = dnnl_primitive_desc_query_md(fwd_pd, dnnl_query_src_md, 0);
    CHECK_DNNL(ai_dnnl_reorder_create(&l->fwd_reorder_src, l->hdr.src_mem, fwd_src_md));
    l->fwd_src_mem = l->fwd_reorder_src.dst_mem;
    // 4. Set up reorder between weights and fwd_weights memory
    const dnnl_memory_desc_t* fwd_weights_md = dnnl_primitive_desc_query_md(fwd_pd, dnnl_query_weights_md, 0);
    CHECK_DNNL(ai_dnnl_reorder_create(&l->fwd_reorder_weights, l->weights_mem, fwd_weights_md));
    l->fwd_weights_mem = l->fwd_reorder_weights.dst_mem;

    // 5. Create output memory
    const dnnl_memory_desc_t* fwd_dst_md = dnnl_primitive_desc_query_md(fwd_pd, dnnl_query_dst_md, 0);
    CHECK_DNNL(dnnl_memory_create(&l->hdr.dst_mem, fwd_dst_md, l->hdr.engine, DNNL_MEMORY_ALLOCATE));
    // 6. Create workspace memory
    const dnnl_memory_desc_t* workspace_md = dnnl_primitive_desc_query_md(fwd_pd, dnnl_query_workspace_md, 0);
    CHECK_DNNL(dnnl_memory_create(&l->workspace_mem, workspace_md, l->hdr.engine, DNNL_MEMORY_ALLOCATE));

    // 7. Clean up
    CHECK_DNNL(dnnl_primitive_desc_destroy(fwd_pd));

    return 0;
dnnl_error:
    return 1;
}


uint32_t conv_layer_bwd_pass_init(ai_dnnl_layer_t* layer, ai_dnnl_layer_t* next_layer)
{
    ai_dnnl_convolutional_layer_t* l = (ai_dnnl_convolutional_layer_t*)layer;

    l->hdr.diff_dst_mem = next_layer->diff_src_mem;

    const_dnnl_primitive_desc_t fwd_pd = ai_dnnl_primitive_get_primitive_desc(l->fwd);

    // 1. bwd data

    // 1.1 Create a convolution bwd data primitive
    // 1.1.1 Create memory descs for involved memory
    const dnnl_memory_desc_t* src_md = ai_dnnl_memory_get_memory_desc(l->hdr.src_mem);
    const dnnl_memory_desc_t* weights_md = ai_dnnl_memory_get_memory_desc(l->weights_mem);
    
    dnnl_dims_t diff_dst_dims = { l->hdr.N, l->hdr.OC, l->hdr.OH, l->hdr.OW };

    dnnl_dims_t strides = { l->SH, l->SW };
    dnnl_dims_t padding_l = { l->PT, l->PL };
    dnnl_dims_t padding_r = { l->PB, l->PR };

    dnnl_memory_desc_t bwd_data_diff_src_md_any;
    dnnl_memory_desc_t bwd_data_weights_md_any;
    dnnl_memory_desc_t bwd_data_diff_dst_md_any;
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&bwd_data_diff_src_md_any, src_md->ndims, src_md->dims, dnnl_f32, dnnl_format_tag_any));
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&bwd_data_diff_dst_md_any, 4, diff_dst_dims, dnnl_f32, dnnl_format_tag_any));
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&bwd_data_weights_md_any, weights_md->ndims, weights_md->dims, dnnl_f32, dnnl_format_tag_any));
    // 1.1.2 Create a convolution bwd data primitive
    dnnl_convolution_desc_t bwd_data_d;
    dnnl_primitive_desc_t bwd_data_pd;
    CHECK_DNNL(dnnl_convolution_backward_data_desc_init(&bwd_data_d, dnnl_convolution_auto, &bwd_data_diff_src_md_any, &bwd_data_weights_md_any, &bwd_data_diff_dst_md_any, strides, padding_l, padding_r));
    CHECK_DNNL(dnnl_primitive_desc_create(&bwd_data_pd, &bwd_data_d, 0, l->hdr.engine, fwd_pd));
    CHECK_DNNL(dnnl_primitive_create(&l->bwd_data, bwd_data_pd));

    // 1.2 Set up reorder between diff_dst and bwd_data_diff_dst
    const dnnl_memory_desc_t* bwd_data_diff_dst_md = dnnl_primitive_desc_query_md(bwd_data_pd, dnnl_query_diff_dst_md, 0);
    CHECK_DNNL(ai_dnnl_reorder_create(&l->bwd_data_reorder_diff_dst, l->hdr.diff_dst_mem, bwd_data_diff_dst_md));
    l->bwd_data_diff_dst_mem = l->bwd_data_reorder_diff_dst.dst_mem;

    // 1.3 Set up reorder between weights and bwd_data_weights
    const dnnl_memory_desc_t* bwd_data_weights_md = dnnl_primitive_desc_query_md(bwd_data_pd, dnnl_query_weights_md, 0);
    CHECK_DNNL(ai_dnnl_reorder_create(&l->bwd_data_reorder_weights, l->fwd_weights_mem, bwd_data_weights_md));
    l->bwd_data_weights_mem = l->bwd_data_reorder_weights.dst_mem;

    // 1.4 Create diff_src mem
    const dnnl_memory_desc_t* bwd_data_diff_src_md = dnnl_primitive_desc_query_md(bwd_data_pd, dnnl_query_diff_src_md, 0);
    CHECK_DNNL(dnnl_memory_create(&l->hdr.diff_src_mem, bwd_data_diff_src_md, l->hdr.engine, DNNL_MEMORY_ALLOCATE));

    // 2. bwd weights

    // 2.1 Create a convolution bwd weights primitive
    // 2.1.1 Create memory descs
    const dnnl_memory_desc_t* dst_md = dnnl_primitive_desc_query_md(fwd_pd, dnnl_query_dst_md, 0);
    dnnl_memory_desc_t bwd_weights_src_md_any;
    dnnl_memory_desc_t bwd_weights_diff_weights_md_any;
    dnnl_memory_desc_t bwd_weights_diff_bias_md_any;
    dnnl_memory_desc_t bwd_weights_diff_dst_md_any;
    dnnl_dims_t bias_dims = { l->hdr.OC };
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&bwd_weights_src_md_any, src_md->ndims, src_md->dims, dnnl_f32, dnnl_format_tag_any));
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&bwd_weights_diff_weights_md_any, weights_md->ndims, weights_md->dims, dnnl_f32, dnnl_format_tag_any));
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&bwd_weights_diff_bias_md_any, 1, bias_dims, dnnl_f32, dnnl_a));
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&bwd_weights_diff_dst_md_any, dst_md->ndims, dst_md->dims, dnnl_f32, dnnl_format_tag_any));
    // 2.1.2 Create a convolution bwd weights primitive
    dnnl_convolution_desc_t bwd_weights_d;
    dnnl_primitive_desc_t bwd_weights_pd;
    CHECK_DNNL(dnnl_convolution_backward_weights_desc_init(&bwd_weights_d, dnnl_convolution_auto, &bwd_weights_src_md_any, &bwd_weights_diff_weights_md_any, &bwd_weights_diff_bias_md_any, &bwd_weights_diff_dst_md_any, strides, padding_l, padding_r));
    CHECK_DNNL(dnnl_primitive_desc_create(&bwd_weights_pd, &bwd_weights_d, 0, l->hdr.engine, fwd_pd));
    CHECK_DNNL(dnnl_primitive_create(&l->bwd_weights, bwd_weights_pd));

    // 2.2 Create diff weights/bias mem
    const dnnl_memory_desc_t* bwd_weights_diff_weights_md = dnnl_primitive_desc_query_md(bwd_weights_pd, dnnl_query_diff_weights_md, 0);
    const dnnl_memory_desc_t* bwd_weights_diff_bias_md = ai_dnnl_memory_get_memory_desc(l->bias_mem);
    CHECK_DNNL(dnnl_memory_create(&l->bwd_weights_diff_weights_mem, bwd_weights_diff_weights_md, l->hdr.engine, DNNL_MEMORY_ALLOCATE));
    CHECK_DNNL(dnnl_memory_create(&l->diff_bias_mem, bwd_weights_diff_bias_md, l->hdr.engine, DNNL_MEMORY_ALLOCATE));

    // 2.3 Set up reorder between fwd_src and bwd_weights_src
    const dnnl_memory_desc_t* bwd_weights_src_md = dnnl_primitive_desc_query_md(bwd_weights_pd, dnnl_query_src_md, 0);
    CHECK_DNNL(ai_dnnl_reorder_create(&l->bwd_weights_reorder_src, l->fwd_src_mem, bwd_weights_src_md));
    l->bwd_weights_src_mem = l->bwd_weights_reorder_src.dst_mem;

    // 2.4 Set up reorder between diff_dst and bwd_weights_diff_dst
    const dnnl_memory_desc_t* bwd_weights_diff_dst_md = dnnl_primitive_desc_query_md(bwd_weights_pd, dnnl_query_diff_dst_md, 0);
    CHECK_DNNL(ai_dnnl_reorder_create(&l->bwd_weights_reorder_diff_dst, l->hdr.diff_dst_mem, bwd_weights_diff_dst_md));
    l->bwd_weights_diff_dst_mem = l->bwd_weights_reorder_diff_dst.dst_mem;

    // 2.5 Set up reorder between bwd_weights_diff_weights and diff_weights
    CHECK_DNNL(ai_dnnl_reorder_create(&l->bwd_weights_reorder_diff_weights, l->bwd_weights_diff_weights_mem, weights_md));
    l->diff_weights_mem = l->bwd_weights_reorder_diff_weights.dst_mem;

    // 3. Clean up
    CHECK_DNNL(dnnl_primitive_desc_destroy(bwd_data_pd));
    CHECK_DNNL(dnnl_primitive_desc_destroy(bwd_weights_pd));
    
    return 0;
dnnl_error:
    return 1;
}

static uint32_t conv_layer_fwd(ai_dnnl_layer_t* layer)
{
    ai_dnnl_convolutional_layer_t* l = (ai_dnnl_convolutional_layer_t*)layer;

    // 1. Reorder src
    CHECK_DNNL(ai_dnnl_reorder_execute(&l->fwd_reorder_src, l->hdr.stream));
    // 2. Reorder weights
    CHECK_DNNL(ai_dnnl_reorder_execute(&l->fwd_reorder_weights, l->hdr.stream));

    // 3. Execute the convolution fwd primitive
    dnnl_exec_arg_t fwd_exec_args[] = {
        { DNNL_ARG_SRC, l->fwd_src_mem },
        { DNNL_ARG_WEIGHTS, l->fwd_weights_mem },
        { DNNL_ARG_BIAS, l->bias_mem },
        { DNNL_ARG_DST, l->hdr.dst_mem },
        { DNNL_ARG_WORKSPACE, l->workspace_mem },
    };
    CHECK_DNNL(dnnl_primitive_execute(l->fwd, l->hdr.stream, 5, fwd_exec_args));

    CHECK_DNNL(dnnl_stream_wait(l->hdr.stream));

    return 0;
dnnl_error:
    return 1;
}


static uint32_t conv_layer_bwd(ai_dnnl_layer_t* layer)
{
    ai_dnnl_convolutional_layer_t* l = (ai_dnnl_convolutional_layer_t*)layer;

    // 1. bwd data

    // 1.1 Reorder weights
    CHECK_DNNL(ai_dnnl_reorder_execute(&l->bwd_data_reorder_weights, l->hdr.stream));
    // 1.2 Reorder diff dst
    CHECK_DNNL(ai_dnnl_reorder_execute(&l->bwd_data_reorder_diff_dst, l->hdr.stream));

    // 1.3 Execute the convolution bwd data primitve
    dnnl_exec_arg_t bwd_data_exec_args[] = {
        { DNNL_ARG_DIFF_SRC, l->hdr.diff_src_mem },
        { DNNL_ARG_WEIGHTS, l->bwd_data_weights_mem },
        { DNNL_ARG_DIFF_DST, l->bwd_data_diff_dst_mem },
        { DNNL_ARG_WORKSPACE, l->workspace_mem }
    };
    CHECK_DNNL(dnnl_primitive_execute(l->bwd_data, l->hdr.stream, 4, bwd_data_exec_args));

    // 2. bwd weights

    // 2.1 Reorder src
    CHECK_DNNL(ai_dnnl_reorder_execute(&l->bwd_weights_reorder_src, l->hdr.stream));
    // 2.2 Reorder diff dst
    CHECK_DNNL(ai_dnnl_reorder_execute(&l->bwd_weights_reorder_diff_dst, l->hdr.stream));

    // 2.3 Execute the convolution bwd weights primitive
    dnnl_exec_arg_t bwd_weights_exec_args[] = {
        { DNNL_ARG_SRC, l->bwd_weights_src_mem },
        { DNNL_ARG_DIFF_WEIGHTS, l->bwd_weights_diff_weights_mem },
        { DNNL_ARG_DIFF_BIAS, l->diff_bias_mem },
        { DNNL_ARG_DIFF_DST, l->bwd_weights_diff_dst_mem },
        { DNNL_ARG_WORKSPACE, l->workspace_mem }
    };
    CHECK_DNNL(dnnl_primitive_execute(l->bwd_weights, l->hdr.stream, 5, bwd_weights_exec_args));

    // 2.4 Reorder diff weights
    CHECK_DNNL(ai_dnnl_reorder_execute(&l->bwd_weights_reorder_diff_weights, l->hdr.stream));

    CHECK_DNNL(dnnl_stream_wait(l->hdr.stream));

    // 2.5 Update weights
    float* weights = ai_dnnl_memory_get_data_handle(l->weights_mem);
    float* diff_weights = ai_dnnl_memory_get_data_handle(l->diff_weights_mem);
    size_t weights_size = l->hdr.OC * l->hdr.IC * l->KH * l->KW;
    for (size_t i = 0; i < weights_size; i++)
        weights[i] -= l->learning_rate * diff_weights[i];
    // 2.6 Update bias
    float* bias = ai_dnnl_memory_get_data_handle(l->bias_mem);
    float* diff_bias = ai_dnnl_memory_get_data_handle(l->diff_bias_mem);
    for (size_t i = 0; i < l->hdr.OC; i++)
        bias[i] -= l->learning_rate * diff_bias[i];

    return 0;
dnnl_error:
    return 1;
}


static uint32_t conv_layer_destroy(ai_dnnl_layer_t* layer)
{
    ai_dnnl_convolutional_layer_t* l = (ai_dnnl_convolutional_layer_t*)layer;

    CHECK_DNNL(dnnl_primitive_destroy(l->fwd));
    CHECK_DNNL(dnnl_primitive_destroy(l->bwd_data));
    CHECK_DNNL(dnnl_primitive_destroy(l->bwd_weights));

    CHECK_DNNL(dnnl_memory_destroy(l->hdr.dst_mem));
    CHECK_DNNL(dnnl_memory_destroy(l->hdr.diff_src_mem));

    CHECK_DNNL(dnnl_memory_destroy(l->weights_mem));
    CHECK_DNNL(dnnl_memory_destroy(l->bias_mem));
    CHECK_DNNL(dnnl_memory_destroy(l->workspace_mem));
    CHECK_DNNL(dnnl_memory_destroy(l->diff_bias_mem));

    CHECK_DNNL(dnnl_memory_destroy(l->bwd_weights_diff_weights_mem));

    CHECK_DNNL(ai_dnnl_reorder_destroy(&l->fwd_reorder_src));
    CHECK_DNNL(ai_dnnl_reorder_destroy(&l->fwd_reorder_weights));
    CHECK_DNNL(ai_dnnl_reorder_destroy(&l->bwd_data_reorder_weights));
    CHECK_DNNL(ai_dnnl_reorder_destroy(&l->bwd_data_reorder_diff_dst));
    CHECK_DNNL(ai_dnnl_reorder_destroy(&l->bwd_weights_reorder_src));
    CHECK_DNNL(ai_dnnl_reorder_destroy(&l->bwd_weights_reorder_diff_weights));
    CHECK_DNNL(ai_dnnl_reorder_destroy(&l->bwd_weights_reorder_diff_dst));

    free(l);

    return 0;
dnnl_error:
    return 1;
}

static float conv_weight_init_xavier(size_t KH, size_t KW, size_t IC)
{
    return AI_RandomNormal(0.0f, sqrtf(1.0f / (KH * KW * IC)));
}

static float conv_weight_init_he(size_t KH, size_t KW, size_t IC)
{
    return AI_RandomNormal(0.0f, sqrtf(2.0f / (KH * KW * IC)));
}

static float conv_bias_init_zeros(size_t KH, size_t KW, size_t IC)
{
    return 0.0f;
}
