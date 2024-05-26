
#include "tensor/tensor_impl.h"
#include "tensor/tensor_math.h"
#include "tensor/tensor_math_internal.h"

#include "log.h"

void tensor_scale(tensor_t* v, float f)
{
    if (v->device == device_cpu) {
        tensor_scale_cpu(v, f);
    } else {
#if defined(USE_GPU)
        tensor_scale_gpu(v, f);
#else
        LOG_ERROR("Invalid device\n");
#endif
    }
}


void tensor_add_scalar(tensor_t* v, float f)
{
    if (v->device == device_cpu) {
        tensor_add_scalar_cpu(v, f);
    } else {
#if defined(USE_GPU)
        tensor_add_scalar_gpu(v, f);
#else
        LOG_ERROR("Invalid device\n");
#endif
    }

}


void tensor_eltwise_add(tensor_t* v, const tensor_t* w)
{
    if (v->device == device_cpu && w->device == device_cpu) {
        tensor_eltwise_add_cpu(v, w);
    } else if (v->device == device_gpu && w->device == device_gpu) {
#if defined(USE_GPU)
        tensor_eltwise_add_gpu(v, w);
#else
        LOG_ERROR("Invalid device\n");
#endif
    } else {
        LOG_ERROR("Tensors must be on same device\n");
    }
}

void tensor_eltwise_mul(tensor_t* v, const tensor_t* w)
{
    if (v->device == device_cpu && w->device == device_cpu) {
        tensor_eltwise_mul_cpu(v, w);
    } else if (v->device == device_gpu && w->device == device_gpu) {
#if defined(USE_GPU)
        tensor_eltwise_mul_gpu(v, w);
#else
        LOG_ERROR("Invalid device\n");
#endif
    } else {
        LOG_ERROR("Tensors must be on same device\n");
    }
}

void tensor_scaled_add(tensor_t* v, const tensor_t* w, float f)
{
    if (v->device == device_cpu && w->device == device_cpu) {
        tensor_scaled_add_cpu(v, w, f);
    } else if (v->device == device_gpu && w->device == device_gpu) {
#if defined(USE_GPU)
        tensor_scaled_add_gpu(v, w, f);
#else
        LOG_ERROR("Invalid device\n");
#endif
    } else {
        LOG_ERROR("Tensors must be on same device\n");
    }
}

void tensor_sum(tensor_t* v, const tensor_t* w)
{
    if (v->device == device_cpu && w->device == device_cpu) {
        tensor_sum_cpu(v, w);
    } else if (v->device == device_gpu && w->device == device_gpu) {
#if defined(USE_GPU)
        tensor_sum_gpu(v, w);
#else
        LOG_ERROR("Invalid device\n");
#endif
    } else {
        LOG_ERROR("Tensors must be on same device\n");
    }
}

void tensor_sum_axis(tensor_t* v, const tensor_t* w, int axis)
{   
    size_t strides[TENSOR_MAX_DIMS];

    size_t curr = 0;
    for (int32_t d = w->shape.ndims - 1; d >= 0; d--) {
        size_t dim = tensor_shape_get_dim(&w->shape, d);
        if (dim != 0) {
            if (curr == 0) {
                curr = 1;
            }
            strides[d] = curr;
            curr *= dim;
        }
    }
    size_t size = tensor_size_from_shape(&w->shape);

    size_t inner_stride = strides[axis];
    size_t axis_len = w->shape.dims[axis];
    size_t outer_stride = axis_len * inner_stride;
    size_t outer_len = size / outer_stride;

    if (v->device == device_cpu && w->device == device_cpu) {
        tensor_sum_axis_cpu(v, w, outer_stride, outer_len, axis_len, inner_stride);
    } else if (v->device == device_gpu && w->device == device_gpu) {
#if defined(USE_GPU)
        tensor_sum_axis_gpu(v, w, outer_stride, outer_len, axis_len, inner_stride);
#else
        LOG_ERROR("Invalid device\n");
#endif
    } else {
        LOG_ERROR("Tensors must be on same device\n");
    }
}
