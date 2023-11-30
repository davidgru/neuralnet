#include <malloc.h>
#include <string.h>

#include "log.h"

#include "tensor.h"
#include "tensor_impl.h"


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


uint32_t tensor_set_zero(tensor_t* tensor)
{
    const tensor_shape_t* shape = tensor_get_shape(tensor);
    memset(tensor->data, 0, tensor_size_from_shape(shape) * sizeof(float));
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
