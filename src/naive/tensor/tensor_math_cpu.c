#include "tensor/tensor_math_internal.h"

#include "util/ai_math.h"
#include "tensor/tensor_impl.h"
#include "random.h"


void tensor_scale_cpu(tensor_t* v, float f)
{
    float* data = tensor_get_data(v);
    size_t n = tensor_get_size(v);
    VectorScale(data, f, n);
}

void tensor_add_scalar_cpu(tensor_t* v, float f)
{
    float* data = tensor_get_data(v);
    size_t n = tensor_get_size(v);
    VectorAddScalar(data, f, n);
}

void tensor_eltwise_add_cpu(tensor_t* v, const tensor_t* w)
{
    float* v_data = tensor_get_data(v);
    const float* w_data = tensor_get_data_const(w);
    size_t n = tensor_get_size(v);
    VectorAdd(v_data, w_data, n);
}

void tensor_eltwise_mul_cpu(tensor_t* v, const tensor_t* w)
{
    float* v_data = tensor_get_data(v);
    const float* w_data = tensor_get_data_const(w);
    size_t n = tensor_get_size(v);
    VectorMul(v_data, w_data, n);
}

void tensor_scaled_add_cpu(tensor_t* v, const tensor_t* w, float f)
{
    float* v_data = tensor_get_data(v);
    const float* w_data = tensor_get_data_const(w);
    size_t n = tensor_get_size(v);
    VectorScaledAdd(v_data, w_data, f, n);
}

void tensor_sum_cpu(tensor_t* v, const tensor_t* w)
{
    float* v_data = tensor_get_data(v);
    const float* w_data = tensor_get_data_const(w);
    size_t n = tensor_get_size(w);
    *v_data += Sum(w_data, n);
}

void tensor_sum_axis_cpu(tensor_t* v, const tensor_t* w, size_t outer_stride, size_t outer_len, size_t axis_len, size_t inner_stride)
{
    for (size_t i = 0; i < outer_len; i++) {
        for (int k = 0; k < inner_stride; k++) {
            v->data[i * inner_stride + k] = 0;
            for (int j = 0; j < axis_len; j++) {
                v->data[i * inner_stride + k] += w->data[i * outer_stride + j * inner_stride + k];
            }
        }
    }
}

void tensor_random_mask_cpu(tensor_t* v, float ratio)
{
    const size_t size = tensor_get_size(v);
    for (size_t i = 0; i < size; i++) {
        v->data[i] = (RandomUniform(0.0f, 1.0f) < ratio);
    }
}
