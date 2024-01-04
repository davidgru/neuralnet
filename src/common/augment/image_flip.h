#pragma once


#include <stdbool.h>

#include "augment_pipeline.h"


typedef struct {
    float horizontal_flip_prob;
    float vertical_flip_prob;
} image_flip_config_t;


extern const augment_impl_t aug_image_flip;


void flip_image_inplace(
    float* image,
    size_t channels,
    size_t height,
    size_t width,
    bool flip_horizontal,
    bool flip_vertical
);
