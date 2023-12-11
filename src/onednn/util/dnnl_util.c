
#include "dnnl_util.h"

#include "dnnl.h"

dnnl_status_t dnnl_reorder_primitive_create(dnnl_primitive_t* reorder_primitive, const_dnnl_memory_desc_t src_md, const_dnnl_memory_desc_t dst_md, dnnl_engine_t engine, dnnl_primitive_attr_t attr)
{
    dnnl_status_t status = dnnl_success;

    // Create a reorder primitive descriptor
    dnnl_primitive_desc_t reorder_pd;
    status = dnnl_reorder_primitive_desc_create(&reorder_pd, src_md, engine, dst_md, engine, attr);
    if (status != dnnl_success) {
        return status;
    }
    // Create the reorder primitive
    status = dnnl_primitive_create(reorder_primitive, reorder_pd);
    if (status != dnnl_success) {
        dnnl_primitive_desc_destroy(reorder_pd);
        return status;
    }
    // Destroy the primitive descriptor
    return dnnl_primitive_desc_destroy(reorder_pd);
}


dnnl_status_t dnnl_reorder_set_up(dnnl_primitive_t* reorder_primitive, dnnl_memory_t src_mem, dnnl_memory_t* dst_mem, const_dnnl_memory_desc_t dst_md, bool* need_reorder, dnnl_engine_t engine)
{
    dnnl_status_t status = dnnl_success;

    // Get memory desc of src mem
    const_dnnl_memory_desc_t src_md = nn_dnnl_memory_get_memory_desc(src_mem);
    // Check if reorder is necessary
    *need_reorder = !dnnl_memory_desc_equal(src_md, dst_md);
    if (*need_reorder) {
        // Create reorder primitive and intermediate memory
        status = dnnl_reorder_primitive_create(reorder_primitive, src_md, dst_md, engine, 0);
        if (status != dnnl_success)
            return status;
        status = dnnl_memory_create(dst_mem, dst_md, engine, DNNL_MEMORY_ALLOCATE);
        if (status != dnnl_success)
            return status;
    }
    else
        *dst_mem = src_mem;

    return status;
}


dnnl_status_t dnnl_reorder_primitive_execute(dnnl_primitive_t reorder_primitive, dnnl_memory_t src_mem, dnnl_memory_t dst_mem, dnnl_stream_t stream)
{
    dnnl_status_t status = dnnl_success;
    dnnl_exec_arg_t exec_args[2] = {
        { DNNL_ARG_FROM, src_mem },
        { DNNL_ARG_TO, dst_mem }
    };
    return dnnl_primitive_execute(reorder_primitive, stream, 2, exec_args);
}


dnnl_status_t dnnl_reorder_once(dnnl_memory_t src_mem, dnnl_memory_t dst_mem, dnnl_stream_t stream)
{
    dnnl_status_t status = dnnl_success;

    dnnl_primitive_t reorder;
    const_dnnl_memory_desc_t src_md;
    const_dnnl_memory_desc_t dst_md;
    dnnl_engine_t engine;

    
    // Get the memory descriptors
    status = dnnl_memory_get_memory_desc(src_mem, &src_md);
    if (status != dnnl_success) {
        return status;
    }
    status = dnnl_memory_get_memory_desc(dst_mem, &dst_md);
    if (status != dnnl_success) {
        return status;
    }
    // Get engine
    status = dnnl_memory_get_engine(src_mem, &engine);
    if (status != dnnl_success) {
        return status;
    }
    // Create a reorder primitive
    status = dnnl_reorder_primitive_create(&reorder, src_md, dst_md, engine, 0);
    if (status != dnnl_success) {
        return status;
    }
    // Execute the reorder primitive
    status = dnnl_reorder_primitive_execute(reorder, src_mem, dst_mem, stream);
    if (status != dnnl_success) {
        dnnl_primitive_destroy(reorder);
        return status;
    }
    // Wait for the primitive to finish execution
    status = dnnl_stream_wait(stream);
    if (status != dnnl_success) {
        dnnl_primitive_destroy(reorder);
        return status;
    }
    // Destroy the primitive
    return dnnl_primitive_destroy(reorder);
}


dnnl_status_t dnnl_memory_create_from_dims(dnnl_memory_t* mem, int ndims, dnnl_dims_t dims, dnnl_data_type_t data_type, dnnl_format_tag_t tag, dnnl_engine_t engine)
{
    dnnl_status_t status = dnnl_success;

    // Create a memory descriptor
    dnnl_memory_desc_t md;
    status = dnnl_memory_desc_create_with_tag(&md, ndims, dims, data_type, tag);
    if (status != dnnl_success) {
        return status;
    }
    // Create the memory object
    return dnnl_memory_create(mem, md, engine, DNNL_MEMORY_ALLOCATE);
}

dnnl_status_t dnnl_memory_create_4_dims(dnnl_memory_t* mem, int64_t dim1, int64_t dim2, int64_t dim3, int64_t dim4, dnnl_data_type_t data_type, dnnl_format_tag_t tag, dnnl_engine_t engine)
{
    dnnl_dims_t dims = { dim1, dim2, dim3, dim4 };
    return dnnl_memory_create_from_dims(mem, 4, dims, data_type, tag, engine);
}

dnnl_status_t dnnl_create_memory_from_1_dim(dnnl_memory_t* mem, int64_t dim, dnnl_data_type_t data_type, dnnl_format_tag_t tag, dnnl_engine_t engine)
{
    dnnl_dims_t dims = { dim };
    return dnnl_memory_create_from_dims(mem, 1, dims, data_type, tag, engine);
}



const_dnnl_memory_desc_t nn_dnnl_memory_get_memory_desc(dnnl_memory_t mem)
{
    const_dnnl_memory_desc_t md;
    dnnl_status_t status = dnnl_memory_get_memory_desc(mem, &md);
    if (status != dnnl_success)
        return 0;
    return md;
}

float* nn_dnnl_memory_get_data_handle(dnnl_memory_t mem)
{
    void* handle = 0;
    dnnl_status_t status = dnnl_memory_get_data_handle(mem, &handle);
    if (status != dnnl_success)
        return 0;
    return (float*)handle;
}


const_dnnl_primitive_desc_t nn_dnnl_primitive_get_primitive_desc(dnnl_primitive_t primitive)
{
    const_dnnl_primitive_desc_t pd;
    dnnl_status_t status = dnnl_primitive_get_primitive_desc(primitive, &pd);
    if (status != dnnl_success)
        return 0;
    return pd;
}


int32_t dnnlutil_memory_desc_get_ndims(const_dnnl_memory_desc_t memory_desc)
{
    int32_t ndims;

    dnnl_memory_desc_query(memory_desc, dnnl_query_ndims_s32, &ndims);
    
    return ndims;
}


const dnnl_dims_t* dnnlutil_memory_desc_get_dims(const_dnnl_memory_desc_t memory_desc)
{
    const dnnl_dims_t* dims;

    dnnl_memory_desc_query(memory_desc, dnnl_query_dims, &dims);

    return dims;
}

dnnl_memory_desc_t dnnlutil_memory_desc_tag_any(const_dnnl_memory_desc_t memory_desc)
{
    int32_t ndims = dnnlutil_memory_desc_get_ndims(memory_desc);
    const dnnl_dims_t* dims = dnnlutil_memory_desc_get_dims(memory_desc);

    dnnl_memory_desc_t desc;
    dnnl_memory_desc_create_with_tag(&desc, ndims, *dims, dnnl_f32, dnnl_format_tag_any);

    return desc;
}