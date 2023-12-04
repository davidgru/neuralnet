#pragma once


#include "dataset.h"


typedef enum {
    MNIST_TRAIN_SET,
    MNIST_TEST_SET
} mnist_dataset_kind_t;


typedef struct {
    const char* path;
    size_t padding;
    mnist_dataset_kind_t dataset_kind;
} mnist_create_info_t;


const extern dataset_impl_t mnist_dataset;
