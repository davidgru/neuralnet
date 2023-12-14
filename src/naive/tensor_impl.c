#include <malloc.h>
#include <stdarg.h>
#include <string.h>

#include "log.h"

#include "tensor.h"
#include "tensor_impl.h"


tensor_shape_t make_tensor_shape(size_t ndims, ...)
{
    tensor_shape_t shape = { 0 };
    va_list args;
    va_start(args, ndims);

    for (size_t i = 0; i < ndims; i++) {
        shape.dims[i] = va_arg(args, size_t);
    }

    va_end(args);
    return shape;
}


size_t tensor_shape_get_dim(const tensor_shape_t* shape, size_t dim)
{
    return shape->dims[dim];
}


size_t tensor_size_from_shape(const tensor_shape_t* shape)
{
    size_t size = 0;
    for (size_t i = 0; i < TENSOR_MAX_DIMS; i++) {
        if (shape->dims[i] != 0) {
            if (size == 0) {
                size = shape->dims[i];
            } else {
                size *= shape->dims[i];
            }
        }
    }
    return size;
}


uint32_t tensor_allocate(tensor_t* tensor, const tensor_shape_t* shape)
{
    tensor->shape = *shape;
 
    size_t size = tensor_size_from_shape(shape);
    if (size != 0) {
        tensor->data = (float*)calloc(size, sizeof(float));
        if (tensor->data == NULL) {
            return 1;
        }
    } else {
        tensor->data = NULL;
    }
    
    return 0;
}


uint32_t tensor_from_memory(tensor_t* tensor, const tensor_shape_t* shape, float* mem)
{
    tensor->shape = *shape;
    tensor->data = mem;
    return 0;
}


uint32_t tensor_copy(tensor_t* tensor_to, const tensor_t* tensor_from)
{
    memcpy(tensor_to->data, tensor_from->data,
        tensor_size_from_shape(&tensor_to->shape) * sizeof(float));
}


uint32_t tensor_fill(tensor_t* tensor, float val)
{
    const tensor_shape_t* shape = tensor_get_shape(tensor);
    for (size_t i = 0; i < tensor_size_from_shape(shape); i++) {
        tensor->data[i] = val;
    }
    
    return 0;
}


uint32_t tensor_set_zero(tensor_t* tensor)
{
    const tensor_shape_t* shape = tensor_get_shape(tensor);
    memset(tensor->data, 0, tensor_size_from_shape(shape) * sizeof(float));
    
    return 0;
}


const tensor_shape_t* tensor_get_shape(const tensor_t* tensor)
{
    return &tensor->shape;
}


float* tensor_get_data(tensor_t* tensor)
{
    return tensor->data;
}


const float* tensor_get_data_const(const tensor_t* tensor)
{
    return tensor->data;
}


uint32_t tensor_destory(tensor_t* tensor)
{
    if (tensor->data != NULL) {
        free(tensor->data);
    }
    return 0;
}
