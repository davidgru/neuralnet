#pragma once

#include "dnnl.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct dnnl_reorder_t {

    dnnl_primitive_t primitive;
    dnnl_memory_t src_mem;
    dnnl_memory_t dst_mem;

    bool need_reorder;
    uint8_t dummy[7];

} dnnl_reorder_t;

dnnl_status_t dnnl_reorder_create(dnnl_reorder_t* reorder, dnnl_memory_t src_mem, const dnnl_memory_desc_t* dst_md);
dnnl_status_t dnnl_reorder_execute(dnnl_reorder_t* reorder, dnnl_stream_t stream);
dnnl_status_t dnnl_reorder_destroy(dnnl_reorder_t* reorder);
