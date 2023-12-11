
#include "dnnl_pooling_layer.h"

#include "util/dnnl_util.h"
#include "util/dnnl_reorder.h"
#include "util/dnnl_assert.h"

#include <malloc.h>

typedef struct dnnl_pooling_layer_t {
    dnnl_layer_t hdr;

    size_t KH;
    size_t KW;
    size_t SH;
    size_t SW;
    size_t PT;
    size_t PL;
    size_t PB;
    size_t PR;

    dnnl_pooling_kind_t pooling_kind;
    int32_t dummy;

    // common
    dnnl_memory_t workspace_mem;

    // fwd
    dnnl_primitive_t fwd;
    
    // bwd
    dnnl_primitive_t bwd;
    dnnl_memory_t bwd_diff_dst_mem;
    
    dnnl_reorder_t bwd_reorder_diff_dst;

} dnnl_pooling_layer_t;


static uint32_t pooling_layer_fwd_pass_init(dnnl_layer_t* layer, dnnl_layer_t* prev_layer);
static uint32_t pooling_layer_bwd_pass_init(dnnl_layer_t* layer, dnnl_layer_t* next_layer);
static uint32_t pooling_layer_fwd(dnnl_layer_t* layer);
static uint32_t pooling_layer_bwd(dnnl_layer_t* layer);
static uint32_t pooling_layer_destroy(dnnl_layer_t* layer);


static dnnl_alg_kind_t get_pooling_alg_kind(dnnl_pooling_kind_t pooling_kind)
{
    static dnnl_alg_kind_t t[] = {
        dnnl_pooling_max,
        dnnl_pooling_avg_include_padding,
        dnnl_pooling_avg_exclude_padding,
    };
    return t[pooling_kind];
}


uint32_t dnnl_pooling_layer_create(dnnl_layer_t** layer, void* create_info)
{
    *layer = (dnnl_layer_t*)malloc(sizeof(dnnl_pooling_layer_t));

    dnnl_pooling_layer_t* l = (dnnl_pooling_layer_t*)*layer;
    dnnl_pooling_layer_create_info_t* i = (dnnl_pooling_layer_create_info_t*)create_info;

    l->KH = i->KH;
    l->KW = i->KW;
    l->SH = i->SH;
    l->SW = i->SW;
    l->PT = i->PT;
    l->PL = i->PL;
    l->PB = i->PB;
    l->PR = i->PR;

    l->pooling_kind = i->pooling_kind;

    l->hdr.fwd_pass_init = pooling_layer_fwd_pass_init;
    l->hdr.bwd_pass_init = pooling_layer_bwd_pass_init;
    l->hdr.fwd = pooling_layer_fwd;
    l->hdr.bwd = pooling_layer_bwd;
    l->hdr.destroy = pooling_layer_destroy;

    return 0;
}


static uint32_t pooling_layer_fwd_pass_init(dnnl_layer_t* layer, dnnl_layer_t* prev_layer)
{
    dnnl_pooling_layer_t* l = (dnnl_pooling_layer_t*)layer;

    l->hdr.N = prev_layer->N;
    l->hdr.IC = prev_layer->OC;
    l->hdr.IH = prev_layer->OH;
    l->hdr.IW = prev_layer->OW;
    l->hdr.OC = l->hdr.IC;
    l->hdr.OH = (l->hdr.IH - l->KH + l->PT + l->PB) / l->SH + 1;
    l->hdr.OW = (l->hdr.IW - l->KW + l->PL + l->PR) / l->SW + 1;

    l->hdr.engine = prev_layer->engine;
    l->hdr.stream = prev_layer->stream;

    l->hdr.src_mem = prev_layer->dst_mem;

    // 1. Create a pooling fwd primitive
    const_dnnl_memory_desc_t src_md = nn_dnnl_memory_get_memory_desc(l->hdr.src_mem);
    dnnl_dims_t dst_dims = { l->hdr.N, l->hdr.OC, l->hdr.OH, l->hdr.OW };
    dnnl_memory_desc_t fwd_dst_md_any;
    CHECK_DNNL(dnnl_memory_desc_create_with_tag(&fwd_dst_md_any, 4, dst_dims, dnnl_f32, dnnl_format_tag_any));
    
    dnnl_alg_kind_t alg_kind = get_pooling_alg_kind(l->pooling_kind);
    dnnl_dims_t strides = { l->SH, l->SW };
    dnnl_dims_t kernel = { l->KH, l->KW };
    dnnl_dims_t padding_l = { l->PT, l->PL };
    dnnl_dims_t padding_r = { l->PB, l->PR };
    dnnl_dims_t dilation = { 1, 1 };

    dnnl_primitive_desc_t fwd_pd;
    CHECK_DNNL(dnnl_pooling_forward_primitive_desc_create(&fwd_pd, l->hdr.engine, dnnl_forward_training,
        alg_kind, src_md, fwd_dst_md_any, strides, kernel, dilation, padding_l, padding_r, NULL));
    CHECK_DNNL(dnnl_primitive_create(&l->fwd, fwd_pd));

    // 2. Create dst memory
    const_dnnl_memory_desc_t dst_md = dnnl_primitive_desc_query_md(fwd_pd, dnnl_query_dst_md, 0);
    CHECK_DNNL(dnnl_memory_create(&l->hdr.dst_mem, dst_md, l->hdr.engine, DNNL_MEMORY_ALLOCATE));
    // 3. Create workspace memory
    const_dnnl_memory_desc_t workspace_md = dnnl_primitive_desc_query_md(fwd_pd, dnnl_query_workspace_md, 0);
    CHECK_DNNL(dnnl_memory_create(&l->workspace_mem, workspace_md, l->hdr.engine, DNNL_MEMORY_ALLOCATE));

    // 4. Clean up
    CHECK_DNNL(dnnl_primitive_desc_destroy(fwd_pd));

    return 0;
dnnl_error:
    return 1;
}

static uint32_t pooling_layer_bwd_pass_init(dnnl_layer_t* layer, dnnl_layer_t* next_layer)
{
    dnnl_pooling_layer_t* l = (dnnl_pooling_layer_t*)layer;

    l->hdr.diff_dst_mem = next_layer->diff_src_mem;

    // 1. Create a pooling bwd primitive
    const_dnnl_primitive_desc_t fwd_pd = nn_dnnl_primitive_get_primitive_desc(l->fwd);
    dnnl_alg_kind_t alg_kind = get_pooling_alg_kind(l->pooling_kind);
    
    dnnl_dims_t strides = { l->SH, l->SW };
    dnnl_dims_t kernel = { l->KH, l->KW };
    dnnl_dims_t padding_l = { l->PT, l->PL };
    dnnl_dims_t padding_r = { l->PB, l->PR };
    dnnl_dims_t dilation = { 1, 1 };

    dnnl_memory_desc_t bwd_diff_src_md_any;
    dnnl_memory_desc_t bwd_diff_dst_md_any;
    dnnl_dims_t bwd_diff_src_dims = { l->hdr.N, l->hdr.IC, l->hdr.IH, l->hdr.IW };
    dnnl_dims_t bwd_diff_dst_dims = { l->hdr.N, l->hdr.OC, l->hdr.OH, l->hdr.OW };
    CHECK_DNNL(dnnl_memory_desc_create_with_tag(&bwd_diff_src_md_any, 4, bwd_diff_src_dims, dnnl_f32, dnnl_format_tag_any));
    CHECK_DNNL(dnnl_memory_desc_create_with_tag(&bwd_diff_dst_md_any, 4, bwd_diff_dst_dims, dnnl_f32, dnnl_format_tag_any));
    
    dnnl_primitive_desc_t bwd_pd;
    CHECK_DNNL(dnnl_pooling_backward_primitive_desc_create(&bwd_pd, l->hdr.engine, alg_kind, bwd_diff_src_md_any,
        bwd_diff_dst_md_any, strides, kernel, dilation, padding_l, padding_r, fwd_pd, NULL));
    CHECK_DNNL(dnnl_primitive_create(&l->bwd, bwd_pd));

    // 2. Set up reorder between diff_dst und bwd_diff_dst
    const_dnnl_memory_desc_t bwd_diff_dst_md = dnnl_primitive_desc_query_md(bwd_pd, dnnl_query_diff_dst_md, 0);
    CHECK_DNNL(dnnl_reorder_create(&l->bwd_reorder_diff_dst, l->hdr.diff_dst_mem, bwd_diff_dst_md));
    l->bwd_diff_dst_mem = l->bwd_reorder_diff_dst.dst_mem;

    // 3. Create diff_src memory
    const_dnnl_memory_desc_t diff_src_md = dnnl_primitive_desc_query_md(bwd_pd, dnnl_query_diff_src_md, 0);
    CHECK_DNNL(dnnl_memory_create(&l->hdr.diff_src_mem, diff_src_md, l->hdr.engine, DNNL_MEMORY_ALLOCATE));

    // 4. Clean up
    CHECK_DNNL(dnnl_primitive_desc_destroy(bwd_pd));

    return 0;
dnnl_error:
    return 1;
}


static uint32_t pooling_layer_fwd(dnnl_layer_t* layer)
{
    dnnl_pooling_layer_t* l = (dnnl_pooling_layer_t*)layer;

    // 1. Just execute the pooling fwd primitive
    dnnl_exec_arg_t exec_args[] = {
        { DNNL_ARG_SRC, l->hdr.src_mem },
        { DNNL_ARG_DST, l->hdr.dst_mem },
        { DNNL_ARG_WORKSPACE, l->workspace_mem }
    };
    CHECK_DNNL(dnnl_primitive_execute(l->fwd, l->hdr.stream, 3, exec_args));

    CHECK_DNNL(dnnl_stream_wait(l->hdr.stream));

    return 0;
dnnl_error:
    return 1;
}


static uint32_t pooling_layer_bwd(dnnl_layer_t* layer)
{
    dnnl_pooling_layer_t* l = (dnnl_pooling_layer_t*)layer;

    // 1. Reorder diff dst
    CHECK_DNNL(dnnl_reorder_execute(&l->bwd_reorder_diff_dst, l->hdr.stream));

    // 2. Execute the pooling bwd primitive
    dnnl_exec_arg_t exec_args[] = {
        { DNNL_ARG_DIFF_SRC, l->hdr.diff_src_mem },
        { DNNL_ARG_DIFF_DST, l->bwd_diff_dst_mem },
        { DNNL_ARG_WORKSPACE, l->workspace_mem }
    };
    CHECK_DNNL(dnnl_primitive_execute(l->bwd, l->hdr.stream, 3, exec_args));

    CHECK_DNNL(dnnl_stream_wait(l->hdr.stream));

    return 0;
dnnl_error:
    return 1;
}


static uint32_t pooling_layer_destroy(dnnl_layer_t* layer)
{
    dnnl_pooling_layer_t* l = (dnnl_pooling_layer_t*)layer;

    CHECK_DNNL(dnnl_primitive_destroy(l->fwd));
    CHECK_DNNL(dnnl_primitive_destroy(l->bwd));

    CHECK_DNNL(dnnl_memory_destroy(l->hdr.dst_mem));
    CHECK_DNNL(dnnl_memory_destroy(l->hdr.diff_src_mem));

    CHECK_DNNL(dnnl_memory_destroy(l->workspace_mem));

    CHECK_DNNL(dnnl_reorder_destroy(&l->bwd_reorder_diff_dst));

    free(l);

    return 0;
dnnl_error:
    return 1;
}
