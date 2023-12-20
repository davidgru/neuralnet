#pragma once


#include "tensor.h"


typedef struct augment_pipeline_s* augment_pipeline_t;


typedef void augment_context_t;
typedef void augment_config_t;

typedef void (*augment_init_func_t)(augment_context_t* context, const augment_config_t* config);
typedef void (*augment_deinit_func_t)(augment_context_t* context);
typedef void (*augment_func_t)(augment_context_t* context, const tensor_t* input, tensor_t* output);
typedef void (*augment_inplace_func_t)(augment_context_t* context, tensor_t* input_output);


typedef struct {
    augment_init_func_t init_func;
    augment_deinit_func_t deinit_func;
    augment_func_t augment_func;
    augment_inplace_func_t augment_inplace_func;
    size_t context_size;
} augment_impl_t;


typedef struct {
    const augment_impl_t* impl;
    const augment_config_t* config;
} augment_pipeline_config_entry_t;


typedef struct {
    const augment_pipeline_config_entry_t* entries;
    size_t num_entries;
} augment_pipeline_config_t;


uint32_t augment_pipeline_create(
    augment_pipeline_t* pipeline,
    const augment_pipeline_config_t* config
);


uint32_t augment_pipeline_forward(
    augment_pipeline_t pipeline,
    const tensor_t* input,
    tensor_t** out_output
);


uint32_t augment_pipeline_destroy(augment_pipeline_t pipeline);
