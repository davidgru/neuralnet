#pragma once


#include "dnnl.h"


#include "tensor.h"


struct tensor {
    tensor_shape_t shape;
    dnnl_memory_t mem;
};
