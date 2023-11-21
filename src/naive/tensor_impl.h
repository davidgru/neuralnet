#pragma once


#include "tensor.h"


struct tensor {
    tensor_shape_t shape;
    float* data;
};
