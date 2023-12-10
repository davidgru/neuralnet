#pragma once

#include <stddef.h>

#include "core/layer.h"
#include "model_desc.h"


typedef struct {
    model_desc_t* desc;
    size_t max_batch_size;
} sequential_model_create_info_t;


const extern layer_impl_t sequential_model_impl;
