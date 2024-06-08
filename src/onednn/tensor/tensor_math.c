
#include "util/ai_math.h"
#include "tensor/tensor_math.h"


void tensor_scaled_add(tensor_t* v, const tensor_t* w, float f)
{
    float* v_data = tensor_get_data(v);
    const float* w_data = tensor_get_data_const(w);
    size_t n = tensor_get_size(v);
    VectorScaledAdd(v_data, w_data, f, n);
}
