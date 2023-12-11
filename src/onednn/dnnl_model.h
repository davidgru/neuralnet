#pragma once

#include "layer/dnnl_base_layer.h"
#include "util/dnnl_loss.h"
#include "model_desc.h"

typedef struct dnnl_model_t {
    dnnl_layer_t* input_layer;
    dnnl_layer_t** layers;
    dnnl_loss_t* loss;
    size_t num_layers;
    
    size_t physical_input_size;
    size_t physical_output_size;
    input_dims_t input_shape;

} dnnl_model_t;

typedef struct dnnl_training_progress_t {
    int32_t epoch;
    float train_loss;
    float train_acc;
    float test_loss;
    float test_acc;
} dnnl_training_progress_t;

typedef void(*dnnl_train_callback_t)(dnnl_training_progress_t* progress_info);

uint32_t dnnl_model_create(dnnl_model_t** model, input_dims_t* input_dims, size_t num_layers, dnnl_layer_create_info_t* layer_create_infos, dnnl_loss_kind_t loss_kind);
uint32_t dnnl_model_create_from_desc(dnnl_model_t** model, model_desc_t* desc);
uint32_t dnnl_model_train(dnnl_model_t* model, size_t train_set_size, float* train_data, uint8_t* train_labels, size_t test_set_size, float* test_data, uint8_t* test_labels, size_t num_epochs, dnnl_train_callback_t callback);
uint32_t dnnl_model_destroy(dnnl_model_t* model);
