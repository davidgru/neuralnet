#pragma once


#include "dnnl.h"

#include "tensor.h"


struct tensor_shape {
    dnnl_memory_desc_t desc;
};


struct tensor {
    tensor_shape_t shape;
    dnnl_memory_t mem;
};


const_dnnl_memory_desc_t memory_desc_from_tensor(const tensor_t* tensor);

uint32_t tensor_from_desc(tensor_t* tensor, const_dnnl_memory_desc_t desc, void* mem);

/* conversion between memory desc and shape */
void shape_from_memory_desc(const_dnnl_memory_desc_t memory_desc);
