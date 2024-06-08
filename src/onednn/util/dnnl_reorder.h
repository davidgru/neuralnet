#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "dnnl.h"

#include "tensor/tensor_impl.h"


typedef struct {
    dnnl_primitive_t primitive;
    tensor_t output;
} dnnl_reorder_t;



dnnl_status_t dnnl_reorder_create(
    dnnl_reorder_t* reorder,
    const_dnnl_memory_desc_t src_md,
    const_dnnl_memory_desc_t dst_md
);


dnnl_status_t dnnl_reorder_execute(
    dnnl_reorder_t* reorder,
    const tensor_t* input,
    const tensor_t** output
);


dnnl_status_t dnnl_reorder_destroy(dnnl_reorder_t* reorder);
