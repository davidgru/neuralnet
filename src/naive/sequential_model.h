#pragma once

#include <stddef.h>

#include "layer/ai_layer.h"
#include "ai_model_desc.h"


typedef struct {
    ai_model_desc_t* desc;
    size_t max_batch_size;
} sequential_model_create_info_t;


const extern layer_impl_t sequential_model_impl;
