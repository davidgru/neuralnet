#pragma once


#include "dataset.h"


typedef struct {
    const char* path;
    size_t padding;
} mnist_create_info_t;


const extern dataset_impl_t mnist_dataset;
