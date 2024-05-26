#pragma once


#include "tensor/tensor.h"


struct tensor_shape {
    /* Dimensions of the tensor. Set unused dimensions to zero. */
    size_t dims[TENSOR_MAX_DIMS];
    size_t ndims;
};


struct tensor {
    struct tensor_shape shape;
    device_t device;
    float* data;
};
