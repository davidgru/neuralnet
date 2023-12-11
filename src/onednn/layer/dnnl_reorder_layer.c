
#include "dnnl_reorder_layer.h"

#include "util/dnnl_util.h"


#include <malloc.h>

#define CHECK_DNNL(call) if (call) goto error

typedef struct dnnl_reorder_layer_t {
    dnnl_layer_t hdr;

    bool fwd_need_reorder;
    bool _dummy[7];

    dnnl_primitive_t fwd_reorder;

} dnnl_reorder_layer_t;

static uint32_t reorder_layer_fwd_pass_init(dnnl_layer_t* layer, dnnl_layer_t* prev_layer);
static uint32_t reorder_layer_bwd_pass_init(dnnl_layer_t* layer, dnnl_layer_t* next_layer);
static uint32_t reorder_layer_fwd(dnnl_layer_t* layer);
static uint32_t reorder_layer_bwd(dnnl_layer_t* layer);
static uint32_t reorder_layer_destroy(dnnl_layer_t* layer);


uint32_t dnnl_reorder_layer_create(dnnl_layer_t** layer, void* create_info)
{
    *layer = (dnnl_layer_t*)malloc(sizeof(dnnl_reorder_layer_t));

    dnnl_reorder_layer_t* l = (dnnl_reorder_layer_t*)*layer;

    l->hdr.fwd_pass_init = reorder_layer_fwd_pass_init;
    l->hdr.bwd_pass_init = reorder_layer_bwd_pass_init;
    l->hdr.fwd = reorder_layer_fwd;
    l->hdr.bwd = reorder_layer_bwd;
    l->hdr.destroy = reorder_layer_destroy;

    return 0;
}

uint32_t reorder_layer_fwd_pass_init(dnnl_layer_t* layer, dnnl_layer_t* prev_layer)
{
    dnnl_reorder_layer_t* l = (dnnl_reorder_layer_t*)layer;

    l->hdr.N = prev_layer->N;
    l->hdr.IC = prev_layer->OC;
    l->hdr.IH = prev_layer->OH;
    l->hdr.IW = prev_layer->OW;

    l->hdr.OC = l->hdr.IC;
    l->hdr.OH = l->hdr.IH;
    l->hdr.OW = l->hdr.IW;

    l->hdr.src_mem = prev_layer->dst_mem;

    l->hdr.engine = prev_layer->engine;
    l->hdr.stream = prev_layer->stream;

    const_dnnl_memory_desc_t src_md = nn_dnnl_memory_get_memory_desc(l->hdr.src_mem);

    // Create dst_md based on src_md
    dnnl_memory_desc_t dst_md = dnnlutil_memory_desc_tag_any(src_md);
    // Set up reorder between src_mem and dst_mem
    CHECK_DNNL(dnnl_reorder_set_up(&l->fwd_reorder, l->hdr.src_mem, &l->hdr.dst_mem, dst_md, &l->fwd_need_reorder, l->hdr.engine));
    CHECK_DNNL(dnnl_memory_create(&l->hdr.diff_src_mem, dst_md, l->hdr.engine, DNNL_MEMORY_ALLOCATE));

    return 0;
error:
    return 1;
}

uint32_t reorder_layer_bwd_pass_init(dnnl_layer_t* layer, dnnl_layer_t* next_layer)
{
    dnnl_reorder_layer_t* l = (dnnl_reorder_layer_t*)layer;
    // Just pass the diff_dst_mem backwards
    l->hdr.diff_dst_mem = l->hdr.diff_src_mem;

    return 0;
}

static uint32_t reorder_layer_fwd(dnnl_layer_t* layer)
{
    dnnl_reorder_layer_t* l = (dnnl_reorder_layer_t*)layer;

    if (l->fwd_need_reorder)
        CHECK_DNNL(dnnl_reorder_primitive_execute(l->fwd_reorder, l->hdr.src_mem, l->hdr.dst_mem, l->hdr.stream));

    return 0;
error:
    return 1;
}

static uint32_t reorder_layer_bwd(dnnl_layer_t* layer)
{
    return 0;
}

static uint32_t reorder_layer_destroy(dnnl_layer_t* layer)
{
    dnnl_reorder_layer_t* l = (dnnl_reorder_layer_t*)layer;

    CHECK_DNNL(dnnl_memory_destroy(l->hdr.diff_src_mem));

    if (l->fwd_need_reorder) {
        CHECK_DNNL(dnnl_primitive_destroy(l->fwd_reorder));
        CHECK_DNNL(dnnl_memory_destroy(l->hdr.dst_mem));
    }

    return 0;
error:
    return 1;
}
