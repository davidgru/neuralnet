#pragma once

#include <stdint.h>

typedef struct AI_MnistDataset{
    float* train_images;
    float* test_images;
    uint8_t* train_labels;
    uint8_t* test_labels;
    size_t num_train_images;
    size_t num_test_images;
    size_t image_width;
    size_t image_height;
} AI_MnistDataset;

// Loads the mnist dataset from the specified path on disk
uint32_t AI_MnistDatasetLoad(AI_MnistDataset* dataset, const char* path, size_t padding);

// Destroys a mnist dataset
void AI_MnistDatasetFree(AI_MnistDataset* dataset);
