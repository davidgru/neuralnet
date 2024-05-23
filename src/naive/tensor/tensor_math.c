
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
