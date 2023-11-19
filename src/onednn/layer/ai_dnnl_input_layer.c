
#include "ai_dnnl_input_layer.h"

#include <malloc.h>

#include <stdio.h>

#define CHECK_DNNL(call)\
    {\
        dnnl_status_t status = call;\
        if (status) { printf("dnnl_error: %d\n", status); goto error; }\
    }

typedef struct ai_dnnl_input_layer_t {
    ai_dnnl_layer_t hdr;
} ai_dnnl_input_layer_t;

static uint32_t input_layer_fwd_pass_init(ai_dnnl_layer_t* layer, ai_dnnl_layer_t* prev_layer);
static uint32_t input_layer_bwd_pass_init(ai_dnnl_layer_t* layer, ai_dnnl_layer_t* next_layer);
static uint32_t input_layer_fwd(ai_dnnl_layer_t* layer);
static uint32_t input_layer_bwd(ai_dnnl_layer_t* layer);
static uint32_t input_layer_destroy(ai_dnnl_layer_t* layer);

uint32_t ai_dnnl_input_layer_create(ai_dnnl_layer_t** layer, void* create_info)
{
    *layer = (ai_dnnl_layer_t*)malloc(sizeof(ai_dnnl_input_layer_t));

    ai_dnnl_input_layer_t* l = (ai_dnnl_input_layer_t*)*layer;

    ai_dnnl_input_layer_create_info_t* info = (ai_dnnl_input_layer_create_info_t*)create_info;

    l->hdr.N = info->N;
    l->hdr.IC = info->C;
    l->hdr.IH = info->H;
    l->hdr.IW = info->W;
    l->hdr.OC = l->hdr.IC;
    l->hdr.OH = l->hdr.IH;
    l->hdr.OW = l->hdr.IW;

    l->hdr.fwd_pass_init = input_layer_fwd_pass_init;
    l->hdr.bwd_pass_init = input_layer_bwd_pass_init;
    l->hdr.fwd = input_layer_fwd;
    l->hdr.bwd = input_layer_bwd;
    l->hdr.destroy = input_layer_destroy;

    l->hdr.engine = info->engine;
    l->hdr.stream = info->stream;

    return 0;
}

static uint32_t input_layer_fwd_pass_init(ai_dnnl_layer_t* layer, ai_dnnl_layer_t* prev_layer)
{
    ai_dnnl_input_layer_t* l = (ai_dnnl_input_layer_t*)layer;

    dnnl_memory_desc_t data_md;
    dnnl_dims_t data_dims = { l->hdr.N, l->hdr.IC, l->hdr.IH, l->hdr.IW };

    CHECK_DNNL(dnnl_memory_desc_init_by_tag(&data_md, 4, data_dims, dnnl_f32, dnnl_nchw));
    CHECK_DNNL(dnnl_memory_create(&l->hdr.dst_mem, &data_md, l->hdr.engine, DNNL_MEMORY_ALLOCATE));

    l->hdr.src_mem = l->hdr.dst_mem;

    return 0;
error:
    return 1;
}

static uint32_t input_layer_bwd_pass_init(ai_dnnl_layer_t* layer, ai_dnnl_layer_t* next_layer)
{
    return 0;
}

static uint32_t input_layer_fwd(ai_dnnl_layer_t* layer)
{
    ai_dnnl_input_layer_t* l = (ai_dnnl_input_layer_t*)layer;

    l->hdr.dst_mem = l->hdr.src_mem;

    return 0;
}

static uint32_t input_layer_bwd(ai_dnnl_layer_t* layer)
{
    return 0;
}

static uint32_t input_layer_destroy(ai_dnnl_layer_t* layer)
{
    ai_dnnl_input_layer_t* l = (ai_dnnl_input_layer_t*)layer;

    CHECK_DNNL(dnnl_memory_destroy(l->hdr.dst_mem));

    free(l);

    return 0;
error:
    return 1;
}
