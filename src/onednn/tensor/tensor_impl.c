#include <string.h>
#include <stdarg.h>

#include "log.h"

#include "tensor/tensor_impl.h"
#include "context_impl.h"


tensor_shape_t make_tensor_shape(size_t ndims, ...)
{
    tensor_shape_t shape = { .desc = NULL };

    va_list args;
    va_start(args, ndims);
    dnnl_dims_t dims = {0};
    for (size_t i = 0; i < ndims; i++) {
        dims[i] = va_arg(args, size_t);
    }
    va_end(args);

    dnnl_format_tag_t tag = packed_format_tag_from_ndims(ndims);
    dnnl_status_t status = dnnl_memory_desc_create_with_tag(&shape.desc, ndims, dims, dnnl_f32, tag);
    if (status != dnnl_success) {
        LOG_ERROR("memory desc create failed with code %d\n", status);
    }

    return shape;
}


tensor_shape_t copy_tensor_shape(const tensor_shape_t* shape)
{
    tensor_shape_t copy = { .desc = NULL };
    
    dnnl_status_t status = dnnl_memory_desc_clone(&copy.desc, shape->desc);
    if (status != dnnl_success) {
        LOG_ERROR("Copying memory desc failed with code %d\n", status);
    }

    return copy;
}


void destroy_tensor_shape(tensor_shape_t* shape)
{
    
    dnnl_memory_desc_destroy(shape->desc);
}


size_t tensor_shape_get_depth_dim(const tensor_shape_t* shape)
{
    int32_t ndims = 0;
    
    dnnl_status_t status = dnnl_memory_desc_query(shape->desc, dnnl_query_ndims_s32, &ndims);
    if (status != dnnl_success) {
        LOG_ERROR("Failed memory desc query with code %d\n", status);
    }
    
    return ndims;
}


size_t tensor_shape_get_dim(const tensor_shape_t* shape, size_t dim)
{
    const dnnl_dims_t* dims;

    dnnl_status_t status = dnnl_memory_desc_query(shape->desc, dnnl_query_dims, &dims);
    if (status != dnnl_success) {
        LOG_ERROR("Failed memory desc query with code %d\n", status);
        return 0;
    }
    
    return (*dims)[dim];
}


size_t tensor_size_from_shape(const tensor_shape_t* shape)
{
    return dnnl_memory_desc_get_size(shape->desc) / dnnl_data_type_size(dnnl_f32);
}


size_t tensor_get_size(const tensor_t* tensor)
{
    return tensor_size_from_shape(&tensor->shape);
}



uint32_t tensor_allocate(tensor_t* tensor, const tensor_shape_t* shape)
{
    return tensor_from_memory(tensor, shape, DNNL_MEMORY_ALLOCATE);
}


uint32_t tensor_from_memory(tensor_t* tensor, const tensor_shape_t* shape, float* mem)
{
    return tensor_from_desc(tensor, shape->desc, mem);
}


uint32_t tensor_from_desc(tensor_t* tensor, const_dnnl_memory_desc_t desc, void* mem)
{
    dnnl_memory_desc_t desc_copy;
    dnnl_memory_desc_clone(&desc_copy, desc);
    tensor->shape.desc = desc_copy;

    dnnl_engine_t eng = get_dnnl_engine();

    dnnl_status_t status = dnnl_memory_create(&tensor->mem, desc, eng, mem);
    if (status != dnnl_success) {
        LOG_ERROR("Creating memory failed with code %d\n", status);
        return 1;
    }

    return 0;
}


uint32_t tensor_copy(tensor_t* tensor_to, const tensor_t* tensor_from)
{
    const_dnnl_memory_desc_t to_md;
    const_dnnl_memory_desc_t from_md;
    dnnl_memory_get_memory_desc(tensor_to->mem, &to_md);
    dnnl_memory_get_memory_desc(tensor_from->mem, &from_md);

    if (dnnl_memory_desc_equal(from_md, to_md) == 0) {
        LOG_ERROR("Can not copy between incompatible memory formats\n");
        return 1;
    }

    float* to_data = tensor_get_data(tensor_to);
    const float* from_data = tensor_get_data_const(tensor_from);
    const size_t size = dnnl_memory_desc_get_size(to_md);


    memcpy(to_data, from_data, size);

    return 0;
}


uint32_t tensor_fill(tensor_t* tensor, float val)
{
    const_dnnl_memory_desc_t md;
    dnnl_memory_get_memory_desc(tensor->mem, &md);
    const size_t nelem = dnnl_memory_desc_get_size(md) / sizeof(float);

    float* data = tensor_get_data(tensor);

    const tensor_shape_t* shape = tensor_get_shape(tensor);
    for (size_t i = 0; i < tensor_size_from_shape(shape); i++) {
        data[i] = val;
    }


    return 0;
}


uint32_t tensor_set_zero(tensor_t* tensor)
{
    return tensor_fill(tensor, 0.0f);
}


const tensor_shape_t* tensor_get_shape(const tensor_t* tensor)
{

    return &tensor->shape;
}


float* tensor_get_data(tensor_t* tensor)
{
    void* data = NULL;
    dnnl_memory_get_data_handle(tensor->mem, &data);
    return data;
}


const float* tensor_get_data_const(const tensor_t* tensor)
{
    void* data = NULL;
    dnnl_memory_get_data_handle(tensor->mem, &data);
    return data;
}


uint32_t tensor_destory(tensor_t* tensor)
{
    dnnl_memory_destroy(tensor->mem);
    return 0;
}


const_dnnl_memory_desc_t memory_desc_from_tensor(const tensor_t* tensor)
{
    const_dnnl_memory_desc_t memory_desc;
    dnnl_memory_get_memory_desc(tensor->mem, &memory_desc);
    return memory_desc;
}


void shape_from_memory_desc(const_dnnl_memory_desc_t memory_desc)
{
    int32_t ndims;
    const dnnl_dims_t* dims;
    if (dnnl_memory_desc_query(memory_desc, dnnl_query_ndims_s32, &ndims) != dnnl_success
        || dnnl_memory_desc_query(memory_desc, dnnl_query_dims, &dims) != dnnl_success) {
        LOG_ERROR("Failed to query memory desc\n");
    }
}


dnnl_format_tag_t packed_format_tag_from_ndims(size_t ndims)
{
    dnnl_format_tag_t tag;

    switch (ndims) {
        case 1: tag = dnnl_a; break;
        case 2: tag = dnnl_ab; break;
        case 3: tag = dnnl_abc; break;
        case 4: tag = dnnl_abcd; break;
        case 5: tag = dnnl_abcde; break;
        case 6: tag = dnnl_abcdef; break;
        case 7: tag = dnnl_abcdefg; break;
        case 8: tag = dnnl_abcdefgh; break;
        case 9: tag = dnnl_abcdefghi; break;
        case 10: tag = dnnl_abcdefghij; break;
        case 11: tag = dnnl_abcdefghijk; break;
        case 12: tag = dnnl_abcdefghijkl; break;
        default: tag = dnnl_format_kind_undef; break;
    }

    return tag;
}
