#include <malloc.h>
#include <stdarg.h>
#include <string.h>

#include "log.h"

#include "tensor/tensor_impl.h"

#if defined(USE_GPU)
#include "_cuda.h"
#endif


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


tensor_shape_t copy_tensor_shape(const tensor_shape_t* shape)
{
    return *shape;
}


void destroy_tensor_shape(tensor_shape_t* shape)
{
    (void)shape;
}


size_t tensor_shape_get_depth_dim(const tensor_shape_t* shape)
{
    return TENSOR_MAX_DIMS;
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
    return tensor_allocate_device(tensor, shape, device_cpu);
}


uint32_t tensor_allocate_device(tensor_t* tensor, const tensor_shape_t* shape, device_t device)
{
    tensor->shape = *shape;
    tensor->device = device;
    tensor->data = NULL;
 
    size_t size = tensor_size_from_shape(shape);
    if (size != 0) {
        if (device == device_cpu) {
            tensor->data = (float*)calloc(size, sizeof(float));
        }
#if defined(USE_GPU)
        else if (device == device_gpu) {
            tensor->data = cuda_malloc(size * sizeof(float));
        }
#endif
    }

    return tensor->data != NULL;
}


uint32_t tensor_from_memory(tensor_t* tensor, const tensor_shape_t* shape, float* mem)
{
    return tensor_from_memory_device(tensor, shape, mem, device_cpu);
}

uint32_t tensor_from_memory_device(tensor_t* tensor, const tensor_shape_t* shape, float* mem, device_t device)
{
    tensor->shape = *shape;
    tensor->device = device;
    tensor->data = mem;
    return 0;
}



uint32_t tensor_copy(tensor_t* tensor_to, const tensor_t* tensor_from)
{
    device_t to_device = tensor_get_device(tensor_to);
    device_t from_device = tensor_get_device(tensor_from);
    size_t to_size = tensor_size_from_shape(&tensor_to->shape) * sizeof(float);
    size_t from_size = tensor_size_from_shape(&tensor_from->shape) * sizeof(float);
    uint32_t err = 0;

    if(to_size != from_size) {
        LOG_ERROR("Tensor size mismatch\n");
        return 1;
    }

    if (from_device == device_cpu && to_device == device_cpu) {
        memcpy(tensor_to->data, tensor_from->data, to_size);
    } else {
#if defined(USE_GPU)
        cuda_memcpy_kind_t kind;
        if (from_device == device_cpu && to_device == device_gpu) {
            kind = cuda_memcpy_host_to_device;
        } else if (from_device == device_gpu && to_device == device_cpu) {
            kind = cuda_memcpy_device_to_host;
        } else if (from_device == device_gpu && to_device == device_gpu) {
            kind = cuda_memcpy_device_to_device;
        }
        err = cuda_memcpy(tensor_to->data, tensor_from->data, to_size, kind);
#else
        err = 1;
#endif
    }
    return err;
}


uint32_t tensor_fill(tensor_t* tensor, float val)
{
    const tensor_shape_t* shape = tensor_get_shape(tensor);
    uint32_t err = 0;

    if (tensor->device == device_cpu) {
        for (size_t i = 0; i < tensor_size_from_shape(shape); i++) {
            tensor->data[i] = val;
        }
    } else {
        tensor_t host_tensor;
        tensor_allocate(&host_tensor, shape);
        tensor_fill(&host_tensor, val); /* this time on cpu */
        err = tensor_copy(tensor, &host_tensor);
        tensor_destory(&host_tensor);
    }
    
    return err;
}


uint32_t tensor_set_zero(tensor_t* tensor)
{
    const tensor_shape_t* shape = tensor_get_shape(tensor);
    uint32_t err = 0;

    if (tensor->device == device_cpu) {
        memset(tensor->data, 0, tensor_size_from_shape(shape) * sizeof(float));
    } else if (tensor->device == device_gpu) {
#if defined(USE_GPU)
        err = cuda_memset(tensor->data, 0, tensor_size_from_shape(shape) * sizeof(float));
#else
        err = 1;
#endif
    } else {
        err = 1;
    }
    
    return err;
}


const tensor_shape_t* tensor_get_shape(const tensor_t* tensor)
{
    return &tensor->shape;
}


size_t tensor_get_size(const tensor_t* tensor)
{
    return tensor_size_from_shape(&tensor->shape);
}


device_t tensor_get_device(const tensor_t* tensor)
{
    return tensor->device;
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
        if (tensor->device == device_cpu) {
            free(tensor->data);
        } else {
#if defined(USE_GPU)
            cuda_free(tensor->data);
#endif
        }
    }
    return 0;
}
