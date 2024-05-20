#pragma once


#include "tensor.h"


struct tensor_shape {
    /* Dimensions of the tensor. Set unused dimensions to zero. */
    size_t dims[TENSOR_MAX_DIMS];
};


struct tensor {
    struct tensor_shape shape;
    device_t device;
    float* data;
};
