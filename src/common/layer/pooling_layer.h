#pragma once

#include "core/layer.h"


typedef enum {
    POOLING_AVERAGE,
    POOLING_MAX,
    POOLING_MIN,
} pooling_kind_t;

typedef struct {
    size_t kernel_width;
    pooling_kind_t pooling_operation;
} pooling_layer_create_info_t;


extern const layer_impl_t pooling_layer_impl;
