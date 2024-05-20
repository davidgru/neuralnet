#include "tensor_math_internal.h"

#include "util/ai_math.h"


void tensor_scale_cpu(tensor_t* v, float f)
{
    float* data = tensor_get_data(v);
    size_t n = tensor_get_size(v);
    VectorScale(data, f, n);
}

void tensor_eltwise_add_cpu(tensor_t* v, const tensor_t* w)
{
    float* v_data = tensor_get_data(v);
    const float* w_data = tensor_get_data_const(w);
    size_t n = tensor_get_size(v);
    VectorAdd(v_data, w_data, n);
}
