#pragma once


#include "dnnl.h"

#include "tensor.h"


struct tensor_shape {
    dnnl_dims_t dims;
    size_t ndims;
    dnnl_format_tag_t tag;
};


struct tensor {
    tensor_shape_t shape;
    dnnl_memory_t mem;
};
