#pragma once


#include <stdbool.h>

#include "augment_pipeline.h"


typedef struct {
    float brightness_augment_prob;
    float brightness_std;
    float contrast_augment_prob;
    float contrast_midpoint;
    float contrast_std;
} color_augment_config_t;


extern const augment_impl_t aug_color;
