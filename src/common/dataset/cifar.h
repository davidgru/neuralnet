#pragma once


#include "dataset.h"


typedef struct {
    const char* path;
} cifar_create_info_t;


const extern dataset_impl_t cifar_dataset;
