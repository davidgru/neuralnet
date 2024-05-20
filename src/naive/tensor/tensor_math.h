#pragma once

#include "tensor.h"

/* v *= f */
void tensor_scale(tensor_t* v, float f);

/* v += w */
void tensor_eltwise_add(tensor_t* v, const tensor_t* w);

