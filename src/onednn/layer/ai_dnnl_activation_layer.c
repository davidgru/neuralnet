

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

#include "../ai_util/ai_dnnl_util.h"
#include "../ai_util/ai_dnnl_reorder.h"
#include "../ai_util/ai_dnnl_assert.h"

#include "ai_dnnl_activation_layer.h"

#include <malloc.h>
#include <stdio.h>

typedef struct ai_dnnl_activation_layer_t {
    
    ai_dnnl_layer_t hdr;

    ai_dnnl_activation_kind_t activation;
    uint32_t dummy;
    float alpha;
    float beta;

    /* oneDNN specific attributes */

    dnnl_primitive_t fwd;
    dnnl_primitive_t bwd;
    
    dnnl_memory_t bwd_diff_dst_mem;

    ai_dnnl_reorder_t bwd_reorder_diff_dst;

} ai_dnnl_activation_layer_t;

// Utility function, converts an ai_dnnl_activation_kind_t to a dnnl_alg_kind_t
static dnnl_alg_kind_t activation_to_alg_kind(ai_dnnl_activation_kind_t activation_kind)
{
    static dnnl_alg_kind_t table[] = {
        dnnl_eltwise_relu_use_dst_for_bwd,
        dnnl_eltwise_tanh_use_dst_for_bwd,
        dnnl_eltwise_logistic_use_dst_for_bwd
    };
    return table[activation_kind];
}

static uint32_t activation_layer_fwd_pass_init(ai_dnnl_layer_t* layer, ai_dnnl_layer_t* prev_layer);
static uint32_t activation_layer_bwd_pass_init(ai_dnnl_layer_t* layer, ai_dnnl_layer_t* next_layer);
static uint32_t activation_layer_fwd(ai_dnnl_layer_t* layer);
static uint32_t activation_layer_bwd(ai_dnnl_layer_t* layer);
static uint32_t activation_layer_destroy(ai_dnnl_layer_t* layer);


uint32_t ai_dnnl_activation_layer_create(ai_dnnl_layer_t** layer, void* create_info)
{
    *layer = (ai_dnnl_layer_t*)malloc(sizeof(ai_dnnl_activation_layer_t));
    
    ai_dnnl_activation_layer_t* l = (ai_dnnl_activation_layer_t*)*layer;
    ai_dnnl_activation_layer_create_info_t* i  = (ai_dnnl_activation_layer_create_info_t*)create_info;

    l->activation = i->activation;
    l->alpha = i->alpha;
    l->beta = i->beta;

    l->hdr.fwd_pass_init = activation_layer_fwd_pass_init;
    l->hdr.bwd_pass_init = activation_layer_bwd_pass_init;
    l->hdr.fwd = activation_layer_fwd;
    l->hdr.bwd = activation_layer_bwd;
    l->hdr.destroy = activation_layer_destroy;
    
    return 0;
}


static uint32_t activation_layer_fwd_pass_init(ai_dnnl_layer_t* layer, ai_dnnl_layer_t* prev_layer)
{
    ai_dnnl_activation_layer_t* l = (ai_dnnl_activation_layer_t*)layer;

    const size_t N = prev_layer->N;
    const size_t C = prev_layer->OC;
    const size_t H = prev_layer->OH;
    const size_t W = prev_layer->OW;

    dnnl_engine_t engine = prev_layer->engine;
    dnnl_stream_t stream = prev_layer->stream;

    l->hdr.N = N;
    l->hdr.IC = C;
    l->hdr.OC = C;
    l->hdr.IH = H;
    l->hdr.OH = H;
    l->hdr.IW = W;
    l->hdr.OW = W;

    l->hdr.engine = engine;
    l->hdr.stream = stream;

    l->hdr.src_mem = prev_layer->dst_mem;

    dnnl_alg_kind_t alg_kind = activation_to_alg_kind(l->activation);

    // 1. Create an eltwise fwd primitive

    // Note that the src memory shouldn't be reordered
    const dnnl_memory_desc_t* data_md = ai_dnnl_memory_get_memory_desc(l->hdr.src_mem);
    dnnl_eltwise_desc_t fwd_d;
    dnnl_primitive_desc_t fwd_pd;
    CHECK_DNNL(dnnl_eltwise_forward_desc_init(&fwd_d, dnnl_forward_training, alg_kind, data_md, l->alpha, l->beta));
    CHECK_DNNL(dnnl_primitive_desc_create(&fwd_pd, &fwd_d, 0, engine, 0));
    CHECK_DNNL(dnnl_primitive_create(&l->fwd, fwd_pd));

    // 2. Create the dst memory
    const dnnl_memory_desc_t* dst_md = dnnl_primitive_desc_query_md(fwd_pd, dnnl_query_dst_md, 0);
    CHECK_DNNL(dnnl_memory_create(&l->hdr.dst_mem, dst_md, engine, DNNL_MEMORY_ALLOCATE));

    // 3. Clean-up
    CHECK_DNNL(dnnl_primitive_desc_destroy(fwd_pd));

    return 0;
dnnl_error:
    return 1;
}


static uint32_t activation_layer_bwd_pass_init(ai_dnnl_layer_t* layer, ai_dnnl_layer_t* next_layer)
{
    ai_dnnl_activation_layer_t* l = (ai_dnnl_activation_layer_t*)layer;

    const size_t N = l->hdr.N;
    const size_t C = l->hdr.IC;
    const size_t H = l->hdr.IH;
    const size_t W = l->hdr.IW;
    
    l->hdr.diff_dst_mem = next_layer->diff_src_mem;

    dnnl_alg_kind_t alg_kind = activation_to_alg_kind(l->activation);

    // 1. Create an eltwise bwd primitive    
    // 1.1 Create memory desc for bwd diff dst memory
    // Note that the diff data has to be same shape as fwd data
    const dnnl_memory_desc_t* src_md = ai_dnnl_memory_get_memory_desc(l->hdr.src_mem);
    dnnl_memory_desc_t bwd_diff_dst_md_any;
    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&bwd_diff_dst_md_any, src_md->ndims, src_md->dims, dnnl_f32, dnnl_format_tag_any));
    // 1.2 Create an eltwise bwd primitive
    dnnl_eltwise_desc_t bwd_d;
    dnnl_primitive_desc_t bwd_pd;
    CHECK_DNNL(dnnl_eltwise_backward_desc_init(&bwd_d, alg_kind, &bwd_diff_dst_md_any, src_md, l->alpha, l->beta));
    const_dnnl_primitive_desc_t fwd_pd = ai_dnnl_primitive_get_primitive_desc(l->fwd);
    CHECK_DNNL(dnnl_primitive_desc_create(&bwd_pd, &bwd_d, 0, l->hdr.engine, fwd_pd));
    CHECK_DNNL(dnnl_primitive_create(&l->bwd, bwd_pd));

    // 2. Set up reorder of diff dst
    const dnnl_memory_desc_t* bwd_diff_dst_md = dnnl_primitive_desc_query_md(bwd_pd, dnnl_query_diff_dst_md, 0);
    CHECK_DNNL(ai_dnnl_reorder_create(&l->bwd_reorder_diff_dst, l->hdr.diff_dst_mem, bwd_diff_dst_md));
    l->bwd_diff_dst_mem = l->bwd_reorder_diff_dst.dst_mem;
    
    // 3. Create diff src memory with same format as diff dst memory
    CHECK_DNNL(dnnl_memory_create(&l->hdr.diff_src_mem, bwd_diff_dst_md, l->hdr.engine, DNNL_MEMORY_ALLOCATE));

    // 4. Clean-up
    CHECK_DNNL(dnnl_primitive_desc_destroy(bwd_pd));

    return 0;
dnnl_error:
    return 1;
}


static uint32_t activation_layer_fwd(ai_dnnl_layer_t* layer)
{
    ai_dnnl_activation_layer_t* l = (ai_dnnl_activation_layer_t*)layer;

    // 1. Just execute the fwd primitive
    dnnl_exec_arg_t exec_args[2] = {
        { DNNL_ARG_SRC, l->hdr.src_mem },
        { DNNL_ARG_DST, l->hdr.dst_mem }
    };
    CHECK_DNNL(dnnl_primitive_execute(l->fwd, l->hdr.stream, 2, exec_args));

    CHECK_DNNL(dnnl_stream_wait(l->hdr.stream));
    
    return 0;
dnnl_error:
    return 1;
}


static uint32_t activation_layer_bwd(ai_dnnl_layer_t* layer)
{
    ai_dnnl_activation_layer_t* l = (ai_dnnl_activation_layer_t*)layer;

    // 1. Reorder diff dst
    CHECK_DNNL(ai_dnnl_reorder_execute(&l->bwd_reorder_diff_dst, l->hdr.stream));

    // 2. Execute the bwd primitive
    dnnl_exec_arg_t exec_args[] = {
        { DNNL_ARG_DST, l->hdr.dst_mem },
        { DNNL_ARG_DIFF_DST, l->bwd_diff_dst_mem },
        { DNNL_ARG_DIFF_SRC, l->hdr.diff_src_mem },
    };
    CHECK_DNNL(dnnl_primitive_execute(l->bwd, l->hdr.stream, 3, exec_args));

    CHECK_DNNL(dnnl_stream_wait(l->hdr.stream));

    return 0;
dnnl_error:
    return 1;
}


static uint32_t activation_layer_destroy(ai_dnnl_layer_t* layer)
{
    ai_dnnl_activation_layer_t* l = (ai_dnnl_activation_layer_t*)layer;

    CHECK_DNNL(dnnl_primitive_destroy(l->fwd));
    CHECK_DNNL(dnnl_primitive_destroy(l->bwd));

    CHECK_DNNL(dnnl_memory_destroy(l->hdr.dst_mem));
    CHECK_DNNL(dnnl_memory_destroy(l->hdr.diff_src_mem));

    CHECK_DNNL(ai_dnnl_reorder_destroy(&l->bwd_reorder_diff_dst));

    free(l);

    return 0;
dnnl_error:
    return 1;
}

