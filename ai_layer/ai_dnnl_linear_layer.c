

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

#include "ai_dnnl_linear_layer.h"

#include "../ai_util/ai_dnnl_util.h"
#include "../ai_util/ai_dnnl_reorder.h"
#include "../ai_util/ai_dnnl_assert.h"
#include "../ai_util/ai_random.h"

#include <math.h>
#include <malloc.h>

#include <stdio.h>

typedef struct ai_dnnl_linear_layer_t {
    ai_dnnl_layer_t hdr;

    ai_dnnl_linear_layer_weight_init_kind_t weight_init;
    ai_dnnl_linear_layer_bias_init_kind_t bias_init;

    float learning_rate;
    int32_t dummy;

    // common memory
    dnnl_memory_t weights_mem;
    dnnl_memory_t bias_mem;
    dnnl_memory_t workspace_mem;
    dnnl_memory_t diff_weights_mem;
    dnnl_memory_t diff_bias_mem;

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

} ai_dnnl_linear_layer_t;


static uint32_t linear_layer_fwd_pass_init(ai_dnnl_layer_t* layer, ai_dnnl_layer_t* prev_layer);
static uint32_t linear_layer_bwd_pass_init(ai_dnnl_layer_t* layer, ai_dnnl_layer_t* next_layer);
static uint32_t linear_layer_fwd(ai_dnnl_layer_t* layer);
static uint32_t linear_layer_bwd(ai_dnnl_layer_t* layer);
static uint32_t linear_layer_destroy(ai_dnnl_layer_t* layer);


typedef float(*weight_init_func)(size_t OC, size_t IC);
typedef float(*bias_init_func)(size_t OC, size_t IC);

static float weight_init_xavier(size_t OC, size_t IC);
static float weight_init_he(size_t OC, size_t IC);
static float bias_init_zeros(size_t OC, size_t IC);


// Utility function to get a weight init function from an enum value
static weight_init_func get_weight_init_func(ai_dnnl_linear_layer_weight_init_kind_t weight_init)
{
    static weight_init_func t[] = {
        weight_init_xavier,
        weight_init_he
    };
    return t[weight_init];
}

// Utility function to get a bias init function from an enum value
static bias_init_func get_bias_init_func(ai_dnnl_linear_layer_bias_init_kind_t bias_init)
{
    static bias_init_func t[] = {
        bias_init_zeros
    };
    return t[bias_init];
}



uint32_t ai_dnnl_linear_layer_create(ai_dnnl_layer_t** layer, void* create_info)
{
    // 1. Allocate memory for the layer
    *layer = (ai_dnnl_layer_t*)malloc(sizeof(ai_dnnl_linear_layer_t));

    ai_dnnl_linear_layer_t* l = (ai_dnnl_linear_layer_t*)*layer;
    ai_dnnl_linear_layer_create_info_t* linear_create_info = (ai_dnnl_linear_layer_create_info_t*)create_info;

    // 2. Set attributes
    l->hdr.OC = linear_create_info->OC;

    l->hdr.fwd_pass_init = linear_layer_fwd_pass_init;
    l->hdr.bwd_pass_init = linear_layer_bwd_pass_init;
    l->hdr.fwd = linear_layer_fwd;
    l->hdr.bwd = linear_layer_bwd;
    l->hdr.destroy = linear_layer_destroy;

    l->hdr.allow_reorder = linear_create_info->allow_reorder;

    l->learning_rate = linear_create_info->learning_rate;
    l->weight_init = linear_create_info->weight_init;
    l->bias_init = linear_create_info->bias_init;

    return 0;
}


static uint32_t linear_layer_fwd_pass_init(ai_dnnl_layer_t* layer, ai_dnnl_layer_t* prev_layer)
{
    ai_dnnl_linear_layer_t* l = (ai_dnnl_linear_layer_t*)layer;

    const size_t N = prev_layer->N;
    const size_t IC = prev_layer->OC;
    const size_t IH = prev_layer->OH;
    const size_t IW = prev_layer->OW;
    const size_t OC = l->hdr.OC;


    l->hdr.N = N;
    l->hdr.IC = IC;
    l->hdr.IH = IH;
    l->hdr.IW = IW;

    l->hdr.OH = 1;
    l->hdr.OW = 1;

    l->hdr.engine = prev_layer->engine;
    l->hdr.stream = prev_layer->stream;

    l->hdr.src_mem = prev_layer->dst_mem;

    // 1. Create an inner product fwd primitive

    // 1.1 Create memory descs for input and outputs with tag any
    dnnl_memory_desc_t fwd_src_md_any;
    dnnl_memory_desc_t fwd_weights_md_any;
    dnnl_memory_desc_t fwd_dst_md_any;
    dnnl_memory_desc_t bias_md;

    // Note that the src memory can be of shape (N, IC, IH, IW)  or (N, IC)
    // The shape of the weights has to be       (OC, IC, IH, IW) or (OC, IC) respectively
    const dnnl_memory_desc_t* src_md = ai_dnnl_memory_get_memory_desc(l->hdr.src_mem);
    uint32_t weights_ndims = 0;
    dnnl_dims_t weights_dims = { 0 };
    dnnl_format_tag_t src_format = src_md->ndims == 2 ? dnnl_nc : dnnl_nchw;
    dnnl_format_tag_t weights_format;
    if (src_md->ndims == 2) {
        weights_ndims = 2;
        weights_dims[0] = OC;
        weights_dims[1] = IC;
        weights_format = dnnl_oi;
    }
    else {
        weights_ndims = 4;
        weights_dims[0] = OC;
        weights_dims[1] = IC;
        weights_dims[2] = IH;
        weights_dims[3] = IW;
        weights_format = dnnl_oihw;
    }

    dnnl_dims_t bias_dims = { OC };
    dnnl_dims_t fwd_dst_dims = { N, OC };
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&fwd_src_md_any, src_md->ndims, src_md->dims, dnnl_f32, l->hdr.allow_reorder ? dnnl_format_tag_any : src_format));
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&fwd_weights_md_any, weights_ndims, weights_dims, dnnl_f32, l->hdr.allow_reorder ? dnnl_format_tag_any : weights_format));
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&bias_md, 1, bias_dims, dnnl_f32, dnnl_a));
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&fwd_dst_md_any, 2, fwd_dst_dims, dnnl_f32, l->hdr.allow_reorder ? dnnl_format_tag_any : dnnl_nc));
    // 1.2 Create an inner product fwd primitive
    dnnl_inner_product_desc_t fwd_d;
    dnnl_primitive_desc_t fwd_pd;
    CHECK_DNNL(dnnl_inner_product_forward_desc_init(&fwd_d, dnnl_forward_training, &fwd_src_md_any, &fwd_weights_md_any, &bias_md, &fwd_dst_md_any));
    CHECK_DNNL(dnnl_primitive_desc_create(&fwd_pd, &fwd_d, 0, l->hdr.engine, 0));
    CHECK_DNNL(dnnl_primitive_create(&l->fwd, fwd_pd));
    
    // 2. Create weights and bias mem
    dnnl_memory_desc_t weights_md;
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&weights_md, weights_ndims, weights_dims, dnnl_f32, weights_format));
    CHECK_DNNL(dnnl_memory_create(&l->weights_mem, &weights_md, l->hdr.engine, DNNL_MEMORY_ALLOCATE));
    CHECK_DNNL(dnnl_memory_create(&l->bias_mem, &bias_md, l->hdr.engine, DNNL_MEMORY_ALLOCATE));
    // Init weights
    weight_init_func weight_init = get_weight_init_func(l->weight_init);
    float* weights = ai_dnnl_memory_get_data_handle(l->weights_mem);
    for (size_t i = 0; i < OC * IC * IH * IW; i++)
        weights[i] = weight_init(OC, IC);
    // Init bias
    bias_init_func bias_init = get_bias_init_func(l->bias_init);
    float* bias = ai_dnnl_memory_get_data_handle(l->bias_mem);
    for (size_t i = 0; i < OC; i++)
        bias[i] = bias_init(OC, IC);

    // 3. Set up reorder between src and fwd_src memory if necessary
    const dnnl_memory_desc_t* fwd_src_md = dnnl_primitive_desc_query_md(fwd_pd, dnnl_query_src_md, 0);
    CHECK_DNNL(ai_dnnl_reorder_create(&l->fwd_reorder_src, l->hdr.src_mem, fwd_src_md));
    l->fwd_src_mem = l->fwd_reorder_src.dst_mem;
    // 4. Set up reorder between weights and fwd_weights memory if necessary
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

static uint32_t linear_layer_bwd_pass_init(ai_dnnl_layer_t* layer, ai_dnnl_layer_t* next_layer)
{
    ai_dnnl_linear_layer_t* l = (ai_dnnl_linear_layer_t*)layer;

    l->hdr.diff_dst_mem = next_layer->diff_src_mem;

    // 1. bwd data
    
    // 1.1 Create an inner product bwd data primitive
    // 1.1.1 Create memory descs for involved memory
    dnnl_memory_desc_t bwd_data_diff_src_md_any;
    dnnl_memory_desc_t bwd_data_weights_md_any;
    dnnl_memory_desc_t bwd_data_diff_dst_md_any;
    // Note that diff_src_mem needs to have same shape as src_mem
    // Also bwd_data_weights_mem needs to have same shape as weights_mem
    const dnnl_memory_desc_t* src_md = ai_dnnl_memory_get_memory_desc(l->hdr.src_mem);
    const dnnl_memory_desc_t* weights_md = ai_dnnl_memory_get_memory_desc(l->weights_mem);
    dnnl_format_tag_t src_format = src_md->ndims == 2 ? dnnl_nc : dnnl_nchw;
    dnnl_format_tag_t weights_format = src_md->ndims == 2 ? dnnl_oi : dnnl_oihw;
    dnnl_dims_t bwd_data_diff_dst_dims = { l->hdr.N, l->hdr.OC };
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&bwd_data_diff_src_md_any, src_md->ndims, src_md->dims, dnnl_f32, l->hdr.allow_reorder ? dnnl_format_tag_any : src_format));
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&bwd_data_weights_md_any, weights_md->ndims, weights_md->dims, dnnl_f32, l->hdr.allow_reorder ? dnnl_format_tag_any : weights_format));
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&bwd_data_diff_dst_md_any, 2, bwd_data_diff_dst_dims, dnnl_f32, l->hdr.allow_reorder ? dnnl_format_tag_any : dnnl_nc));
    // 1.1.2 Create the primitive
    dnnl_inner_product_desc_t bwd_data_d;
    dnnl_primitive_desc_t bwd_data_pd;
    CHECK_DNNL(dnnl_inner_product_backward_data_desc_init(&bwd_data_d, &bwd_data_diff_src_md_any, &bwd_data_weights_md_any, &bwd_data_diff_dst_md_any));
    const_dnnl_primitive_desc_t fwd_pd = ai_dnnl_primitive_get_primitive_desc(l->fwd);
    CHECK_DNNL(dnnl_primitive_desc_create(&bwd_data_pd, &bwd_data_d, 0, l->hdr.engine, fwd_pd));
    CHECK_DNNL(dnnl_primitive_create(&l->bwd_data, bwd_data_pd));

    // 1.2 Set up reorder between diff_dst and bwd_data_diff_dst if necessary
    const dnnl_memory_desc_t* bwd_data_diff_dst_md = dnnl_primitive_desc_query_md(bwd_data_pd, dnnl_query_diff_dst_md, 0);
    CHECK_DNNL(ai_dnnl_reorder_create(&l->bwd_data_reorder_diff_dst, l->hdr.diff_dst_mem, bwd_data_diff_dst_md));
    l->bwd_data_diff_dst_mem = l->bwd_data_reorder_diff_dst.dst_mem;

    // 1.3 Set up reorder between weights and bwd_data_weights if necessary
    const dnnl_memory_desc_t* bwd_data_weights_md = dnnl_primitive_desc_query_md(bwd_data_pd, dnnl_query_weights_md, 0);
    CHECK_DNNL(ai_dnnl_reorder_create(&l->bwd_data_reorder_weights, l->fwd_weights_mem, bwd_data_weights_md));
    l->bwd_data_weights_mem = l->bwd_data_reorder_weights.dst_mem;

    // 1.4 Create diff_src memory
    const dnnl_memory_desc_t* bwd_data_diff_src_md = dnnl_primitive_desc_query_md(bwd_data_pd, dnnl_query_diff_src_md, 0);
    CHECK_DNNL(dnnl_memory_create(&l->hdr.diff_src_mem, bwd_data_diff_src_md, l->hdr.engine, DNNL_MEMORY_ALLOCATE));

    // 2. bwd weights

    // 2.1 Create an inner product bwd weights primitive
    // 2.1.1 Create memory descs
    dnnl_memory_desc_t bwd_weights_src_md_any;
    dnnl_memory_desc_t bwd_weights_diff_weights_md_any;
    dnnl_memory_desc_t bwd_weights_diff_bias_md_any;
    dnnl_memory_desc_t bwd_weights_diff_dst_md_any;
    // Note that bwd_weights_src_mem needs to have same shape as src_mem
    // Also bwd_weights_diff_weights needs to have same shape as weights_mem
    dnnl_dims_t bwd_weights_diff_bias_dims = { l->hdr.OC };
    dnnl_dims_t bwd_weights_diff_dst_dims = { l->hdr.N, l->hdr.OC };
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&bwd_weights_src_md_any, src_md->ndims, src_md->dims, dnnl_f32, l->hdr.allow_reorder ? dnnl_format_tag_any : src_format));
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&bwd_weights_diff_weights_md_any, weights_md->ndims, weights_md->dims, dnnl_f32, l->hdr.allow_reorder ? dnnl_format_tag_any : weights_format));
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&bwd_weights_diff_bias_md_any, 1, bwd_weights_diff_bias_dims, dnnl_f32, dnnl_a));
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&bwd_weights_diff_dst_md_any, 2, bwd_weights_diff_dst_dims, dnnl_f32, l->hdr.allow_reorder ? dnnl_format_tag_any : dnnl_nc));
    // 2.1.2 Create the primitive
    dnnl_inner_product_desc_t bwd_weights_d;
    dnnl_primitive_desc_t bwd_weights_pd;
    CHECK_DNNL(dnnl_inner_product_backward_weights_desc_init(&bwd_weights_d, &bwd_weights_src_md_any, &bwd_weights_diff_weights_md_any, &bwd_weights_diff_bias_md_any, &bwd_weights_diff_dst_md_any));
    CHECK_DNNL(dnnl_primitive_desc_create(&bwd_weights_pd, &bwd_weights_d, 0, l->hdr.engine, fwd_pd));
    CHECK_DNNL(dnnl_primitive_create(&l->bwd_weights, bwd_weights_pd));

    // 2.2 Create diff weights/bias mem
    const dnnl_memory_desc_t* bwd_weights_diff_weights_md = dnnl_primitive_desc_query_md(bwd_weights_pd, dnnl_query_diff_weights_md, 0);
    const dnnl_memory_desc_t* bwd_weights_diff_bias_md = ai_dnnl_memory_get_memory_desc(l->bias_mem);
    CHECK_DNNL(dnnl_memory_create(&l->bwd_weights_diff_weights_mem, bwd_weights_diff_weights_md, l->hdr.engine, DNNL_MEMORY_ALLOCATE));
    CHECK_DNNL(dnnl_memory_create(&l->diff_bias_mem, bwd_weights_diff_bias_md, l->hdr.engine, DNNL_MEMORY_ALLOCATE));

    // 2.3 Set up reorder between fwd_src_mem and bwd_weights_src_mem
    const dnnl_memory_desc_t* bwd_weights_src_md = dnnl_primitive_desc_query_md(bwd_weights_pd, dnnl_query_src_md, 0);
    CHECK_DNNL(ai_dnnl_reorder_create(&l->bwd_weights_reorder_src, l->fwd_src_mem, bwd_weights_src_md));
    l->bwd_weights_src_mem = l->bwd_weights_reorder_src.dst_mem;

    // 2.4 Set up reorder between diff_dst and bwd_weights_diff_dst
    const dnnl_memory_desc_t* bwd_weights_diff_dst_md = dnnl_primitive_desc_query_md(bwd_weights_pd, dnnl_query_diff_dst_md, 0);
    CHECK_DNNL(ai_dnnl_reorder_create(&l->bwd_weights_reorder_diff_dst, l->hdr.diff_dst_mem, bwd_weights_diff_dst_md));
    l->bwd_weights_diff_dst_mem = l->bwd_weights_reorder_diff_dst.dst_mem;
    // 2.5 Set up reorder between bwd_data_diff_weights and diff_weights
    CHECK_DNNL(ai_dnnl_reorder_create(&l->bwd_weights_reorder_diff_weights, l->bwd_weights_diff_weights_mem, weights_md));
    l->diff_weights_mem = l->bwd_weights_reorder_diff_weights.dst_mem;

    // 3. Clean up
    CHECK_DNNL(dnnl_primitive_desc_destroy(bwd_data_pd));
    CHECK_DNNL(dnnl_primitive_desc_destroy(bwd_weights_pd));

    return 0;
dnnl_error:
    return 1;
}


static uint32_t linear_layer_fwd(ai_dnnl_layer_t* layer)
{
    ai_dnnl_linear_layer_t* l = (ai_dnnl_linear_layer_t*)layer;

    // 1. Reorder src
    CHECK_DNNL(ai_dnnl_reorder_execute(&l->fwd_reorder_src, l->hdr.stream));
    // 2. Reorder weights
    CHECK_DNNL(ai_dnnl_reorder_execute(&l->fwd_reorder_weights, l->hdr.stream));

    // 3. Execute the inner product fwd primitive
    dnnl_exec_arg_t exec_args[] = {
        { DNNL_ARG_SRC, l->fwd_src_mem },
        { DNNL_ARG_WEIGHTS, l->fwd_weights_mem },
        { DNNL_ARG_BIAS, l->bias_mem },
        { DNNL_ARG_DST, l->hdr.dst_mem },
        { DNNL_ARG_WORKSPACE, l->workspace_mem }
    };
    CHECK_DNNL(dnnl_primitive_execute(l->fwd, l->hdr.stream, 5, exec_args));

    CHECK_DNNL(dnnl_stream_wait(l->hdr.stream));

    return 0;
dnnl_error:
    return 1;
}


static uint32_t linear_layer_bwd(ai_dnnl_layer_t* layer)
{
    ai_dnnl_linear_layer_t* l = (ai_dnnl_linear_layer_t*)layer;

    // 1. Bwd data

    // 1.1 Reorder weights
    CHECK_DNNL(ai_dnnl_reorder_execute(&l->bwd_data_reorder_weights, l->hdr.stream));
    // 1.2 Reorder diff dst
    CHECK_DNNL(ai_dnnl_reorder_execute(&l->bwd_data_reorder_diff_dst, l->hdr.stream));

    // 1.3 Execute the inner product bwd data primitive
    dnnl_exec_arg_t data_exec_args[] = {
        { DNNL_ARG_DIFF_SRC, l->hdr.diff_src_mem },
        { DNNL_ARG_WEIGHTS, l->bwd_data_weights_mem },
        { DNNL_ARG_DIFF_DST, l->bwd_data_diff_dst_mem },
        { DNNL_ARG_WORKSPACE, l->workspace_mem }
    };
    CHECK_DNNL(dnnl_primitive_execute(l->bwd_data, l->hdr.stream, 4, data_exec_args));

    // 2. Bwd weights

    // 2.1 Reorder src
    CHECK_DNNL(ai_dnnl_reorder_execute(&l->bwd_weights_reorder_src, l->hdr.stream));
    // 2.2 Reorder diff dst
    CHECK_DNNL(ai_dnnl_reorder_execute(&l->bwd_weights_reorder_diff_dst, l->hdr.stream));

    // 2.3 Execute the inner product bwd weights primitive
    dnnl_exec_arg_t weights_exec_args[] = {
        { DNNL_ARG_SRC, l->bwd_weights_src_mem },
        { DNNL_ARG_DIFF_WEIGHTS, l->bwd_weights_diff_weights_mem },
        { DNNL_ARG_DIFF_BIAS, l->diff_bias_mem },
        { DNNL_ARG_DIFF_DST, l->bwd_weights_diff_dst_mem },
        { DNNL_ARG_WORKSPACE, l->workspace_mem }
    };
    CHECK_DNNL(dnnl_primitive_execute(l->bwd_weights, l->hdr.stream, 5, weights_exec_args));

    // 2.4 Reorder diff weights
    CHECK_DNNL(ai_dnnl_reorder_execute(&l->bwd_weights_reorder_diff_weights, l->hdr.stream));

    CHECK_DNNL(dnnl_stream_wait(l->hdr.stream));

    // 2.5 Update weights
    float* weights = ai_dnnl_memory_get_data_handle(l->weights_mem);
    float* diff_weights = ai_dnnl_memory_get_data_handle(l->diff_weights_mem);
    size_t weights_dims = ai_dnnl_memory_get_memory_desc(l->weights_mem)->ndims;
    size_t weights_size = weights_dims == 2 ? (l->hdr.OC * l->hdr.IC) : (l->hdr.OC * l->hdr.IC * l->hdr.IH * l->hdr.IW);
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


static uint32_t linear_layer_destroy(ai_dnnl_layer_t* layer)
{
    ai_dnnl_linear_layer_t* l = (ai_dnnl_linear_layer_t*)layer;

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
    CHECK_DNNL(ai_dnnl_reorder_destroy(&l->bwd_weights_reorder_diff_dst));
    CHECK_DNNL(ai_dnnl_reorder_destroy(&l->bwd_weights_reorder_diff_weights));

    free(l);

    return 0;
dnnl_error:
    return 1;
}



static float weight_init_xavier(size_t OC, size_t IC)
{
    return AI_RandomNormal(0.0f, sqrtf(1.0f / OC));
}

static float weight_init_he(size_t OC, size_t IC)
{
    return AI_RandomNormal(0.0f, sqrtf(2.0f / OC));
}

static float bias_init_zeros(size_t OC, size_t IC)
{
    return 0.0f;
}
