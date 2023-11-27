#pragma once

#include <stdbool.h>

#include "tensor.h"
#include "tensor_impl.h"

typedef void dataset_context_t;
typedef void dataset_create_info_t;


typedef uint32_t(*dataset_init_func_t)(
    dataset_context_t* context,
    const dataset_create_info_t* create_info,
    tensor_shape_t* out_data_shape
);


typedef uint32_t(*dataset_get_batch_func_t)(
    dataset_context_t* context,
    const size_t* indices,
    tensor_t* out_batch,
    uint8_t* out_labels
);


typedef uint32_t(*dataset_deinit_func_t)(dataset_context_t* context);


typedef struct {
    dataset_init_func_t init_func;
    dataset_get_batch_func_t get_batch_func;
    dataset_deinit_func_t deinit_func;
    size_t conext_size;
} dataset_impl_t;


typedef struct dataset_s* dataset_t;


uint32_t dataset_create(
    dataset_t* dataset,
    const dataset_impl_t* impl,
    const dataset_create_info_t* create_info
);


const tensor_shape_t* dataset_get_shape(dataset_t dataset);


uint32_t dataset_iteration_begin(
    dataset_t dataset,
    size_t batch_size,
    bool shuffle,
    tensor_t** out_batch,
    uint8_t** out_labels
);


uint32_t dataset_iteration_next(dataset_t dataset, tensor_t** out_batch, uint8_t** out_labels);


uint32_t dataset_destroy(dataset_t dataset);