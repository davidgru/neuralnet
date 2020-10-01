#pragma once

#include "dnnl.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

dnnl_status_t ai_dnnl_reorder_primitive_create(dnnl_primitive_t* reorder_primitive, const dnnl_memory_desc_t* src_md, const dnnl_memory_desc_t* dst_md, dnnl_engine_t engine, dnnl_primitive_attr_t attr);
dnnl_status_t ai_dnnl_reorder_set_up(dnnl_primitive_t* reorder_primitive, dnnl_memory_t src_mem, dnnl_memory_t* dst_mem, const dnnl_memory_desc_t* dst_md, bool* need_reorder, dnnl_engine_t engine);
dnnl_status_t ai_dnnl_reorder_primitive_execute(dnnl_primitive_t reorder_primitive, dnnl_memory_t src_mem, dnnl_memory_t dst_mem, dnnl_stream_t stream);
dnnl_status_t ai_dnnl_reorder_once(dnnl_memory_t src_mem, dnnl_memory_t dst_mem, dnnl_stream_t stream);

dnnl_status_t ai_dnnl_memory_create_from_dims(dnnl_memory_t* mem, int ndims, dnnl_dims_t dims, dnnl_data_type_t data_type, dnnl_format_tag_t tag, dnnl_engine_t engine);
dnnl_status_t ai_dnnl_memory_create_4_dims(dnnl_memory_t* mem, int64_t dim1, int64_t dim2, int64_t dim3, int64_t dim4, dnnl_data_type_t data_type, dnnl_format_tag_t tag, dnnl_engine_t engine);
dnnl_status_t ai_dnnl_memory_create_1_dim(dnnl_memory_t* mem, int64_t dim, dnnl_data_type_t data_type, dnnl_format_tag_t tag, dnnl_engine_t engine);

const dnnl_memory_desc_t* ai_dnnl_memory_get_memory_desc(dnnl_memory_t mem);
float* ai_dnnl_memory_get_data_handle(dnnl_memory_t mem);

const_dnnl_primitive_desc_t ai_dnnl_primitive_get_primitive_desc(dnnl_primitive_t primitive);
