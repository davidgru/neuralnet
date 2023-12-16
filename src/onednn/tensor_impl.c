#include <string.h>
#include <stdarg.h>

#include "log.h"

#include "tensor_impl.h"
#include "context_impl.h"

static dnnl_format_tag_t packed_format_tag_from_ndims(size_t ndims);


tensor_shape_t make_tensor_shape(size_t ndims, ...)
{
    tensor_shape_t shape;

    if (ndims > DNNL_MAX_NDIMS) {
        LOG_ERROR("ndims %zu exceeds DNNL_MAX_DIMS=%d\n", ndims, DNNL_MAX_NDIMS);
        return shape;
    }

    shape.ndims = ndims;

    va_list args;
    va_start(args, ndims);
    for (size_t i = 0; i < ndims; i++) {
        shape.dims[i] = va_arg(args, size_t);
    }
    va_end(args);

    shape.tag = packed_format_tag_from_ndims(ndims);

    return shape;
}


size_t tensor_shape_get_dim(const tensor_shape_t* shape, size_t dim)
{
    if (dim >= shape->ndims) {
        LOG_ERROR("dim=%zu out of bounds. ndims is %zu\n", dim, shape->ndims);
        return 0;
    }
    return shape->dims[dim];
}


size_t tensor_size_from_shape(const tensor_shape_t* shape)
{
    if (shape->ndims == 0) {
        return 0;
    }

    size_t size = 1;
    for (size_t i = 0; i < shape->ndims; i++) {
        size *= shape->dims[i];
    }
    return size;
}


uint32_t tensor_allocate(tensor_t* tensor, const tensor_shape_t* shape)
{
    return tensor_from_memory(tensor, shape, DNNL_MEMORY_ALLOCATE);
}


uint32_t tensor_from_memory(tensor_t* tensor, const tensor_shape_t* shape, float* mem)
{
    tensor->shape = *shape;
 
    dnnl_engine_t eng = get_dnnl_engine();

    dnnl_memory_desc_t md;
    const dnnl_dims_t dims = {
        shape->dims[TENSOR_BATCH_DIM],
        shape->dims[TENSOR_CHANNEL_DIM],
        shape->dims[TENSOR_HEIGHT_DIM],
        shape->dims[TENSOR_WIDTH_DIM],
    };
    dnnl_memory_desc_create_with_tag(&md, shape->ndims, shape->dims, dnnl_f32, shape->tag);

    dnnl_memory_create(&tensor->mem, md, eng, mem);

    /* Do not need the memory desc anymore. */
    dnnl_memory_desc_destroy(md);

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
    LOG_WARN("Tensor shape is possibly not initialized\n");

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


static dnnl_format_tag_t packed_format_tag_from_ndims(size_t ndims)
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
