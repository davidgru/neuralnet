
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


typedef struct {
    size_t inner_stride;
    size_t outer_stride;
    size_t axis_len;
    size_t outer_len;
} reduction_strides_t;

static void get_reduction_strides(const tensor_t* w, const tensor_t* mean, int axis, reduction_strides_t* out_strides)
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

    out_strides->inner_stride = strides[axis];
    out_strides->outer_stride = w->shape.dims[axis] * strides[axis];
    out_strides->axis_len = w->shape.dims[axis];
    out_strides->outer_len = size / out_strides->outer_stride;
}

void tensor_sum_axis(tensor_t* v, const tensor_t* w, int axis)
{   
    reduction_strides_t strides = { 0 };
    get_reduction_strides(w, NULL, axis, &strides);

    if (v->device == device_cpu && w->device == device_cpu) {
        tensor_sum_axis_cpu(v, w, strides.outer_stride, strides.outer_len,
                            strides.axis_len, strides.inner_stride);
    } else if (v->device == device_gpu && w->device == device_gpu) {
#if defined(USE_GPU)
        tensor_sum_axis_gpu(v, w, strides.outer_stride, strides.outer_len,
                            strides.axis_len, strides.inner_stride);
#else
        LOG_ERROR("Invalid device\n");
#endif
    } else {
        LOG_ERROR("Tensors must be on same device\n");
    }
}

/* v = mean(w, axis=axis)*/
void tensor_mean_axis(tensor_t* v, const tensor_t* w, int axis)
{
    tensor_sum_axis(v, w, axis);
    tensor_scale(v, 1.0f / w->shape.dims[axis]);
}

/* v = var(w, axis=axis)*/
void tensor_variance_axis(tensor_t* v, const tensor_t* w, const tensor_t* mean, int axis)
{
    reduction_strides_t strides = { 0 };
    get_reduction_strides(w, mean, axis, &strides);

    if (v->device == device_cpu && w->device == device_cpu) {
        tensor_variance_axis_cpu(v, w, mean, strides.outer_stride, strides.outer_len,
                            strides.axis_len, strides.inner_stride);
    } else if (v->device == device_gpu && w->device == device_gpu) {
#if defined(USE_GPU)
        tensor_variance_axis_gpu(v, w, mean, strides.outer_stride, strides.outer_len,
                            strides.axis_len, strides.inner_stride);
#else
        LOG_ERROR("Invalid device\n");
#endif
    } else {
        LOG_ERROR("Tensors must be on same device\n");
    }
}

void tensor_random_mask(tensor_t* v, float ratio)
{
    if (v->device == device_cpu) {
        tensor_random_mask_cpu(v, ratio);
    } else {
#if defined(USE_GPU)
        tensor_random_mask_gpu(v, ratio);
#else
        LOG_ERROR("Invalid device\n");
#endif
    }
}

void tensor_momentum_update(tensor_t* v, const tensor_t* w, float momentum)
{
    if (v->device == device_cpu && w->device == device_cpu) {
        tensor_momentum_update_cpu(v, w, momentum);
    } else if (v->device == device_gpu && w->device == device_gpu) {
#if defined(USE_GPU)
        tensor_momentum_update_gpu(v, w, momentum);
#else
        LOG_ERROR("Invalid device\n");
#endif
    } else {
        LOG_ERROR("Tensors must be on same device\n");
    }
}
