#pragma once


#include "tensor/tensor.h"

/* v *= f */
void tensor_scale(tensor_t* v, float f);

/* v += f */
void tensor_add_scalar(tensor_t* v, float f);

/* v += w */
void tensor_eltwise_add(tensor_t* v, const tensor_t* w);

/* v *= w */
void tensor_eltwise_mul(tensor_t* v, const tensor_t* w);

/* v += f * w */
void tensor_scaled_add(tensor_t* v, const tensor_t* w, float f);

/* v += sum(w) */
void tensor_sum(tensor_t* v, const tensor_t* w);

/* v += sum(w, axis=axis)*/
void tensor_sum_axis(tensor_t* v, const tensor_t* w, int axis);
