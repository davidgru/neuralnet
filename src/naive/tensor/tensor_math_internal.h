#include "tensor_impl.h"

#ifdef __cplusplus
extern "C" {
#endif

void tensor_scale_cpu(tensor_t* v, float f);
void tensor_eltwise_add_cpu(tensor_t* v, const tensor_t* w);

#if defined(USE_GPU)
void tensor_scale_gpu(tensor_t* v, float f);
void tensor_eltwise_add_gpu(tensor_t* v, const tensor_t* w);
#endif

#ifdef __cplusplus
}
#endif
