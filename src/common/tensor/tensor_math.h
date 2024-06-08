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

/* v = sum(w, axis=axis)*/
void tensor_sum_axis(tensor_t* v, const tensor_t* w, int axis);

/* v = mean(w, axis=axis)*/
void tensor_mean_axis(tensor_t* v, const tensor_t* w, int axis);

/* v = var(w, axis=axis)*/
void tensor_variance_axis(tensor_t* v, const tensor_t* w, const tensor_t* mean, int axis);

/* v = random(1 with prob ratio)*/
void tensor_random_mask(tensor_t* v, float ratio);

/* v = (momentum) * v + (1.0 - momentum) * w */
void tensor_momentum_update(tensor_t* v, const tensor_t* w, float momentum);
