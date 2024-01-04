#pragma once


#include <stdbool.h>

#include "augment_pipeline.h"


typedef struct {
    int32_t padding; /* implicit padding applied to the input before cropping */
} random_crop_config_t;


extern const augment_impl_t aug_random_crop;


void crop_image(
    const float* input,
    float* output,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t startY,
    int32_t startX
);
