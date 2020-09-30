#pragma once

#include "ai_layer/ai_dnnl_base_layer.h"
#include "ai_util/ai_dnnl_loss.h"


typedef struct ai_input_dims_t {
    size_t N;
    size_t C;
    size_t H;
    size_t W;
} ai_input_dims_t;

typedef struct ai_dnnl_model_t {
    ai_dnnl_layer_t* input_layer;
    ai_dnnl_layer_t** layers;
    ai_dnnl_loss_t* loss;
    size_t num_layers;
    
    size_t physical_input_size;
    size_t physical_output_size;
    ai_input_dims_t input_shape;

} ai_dnnl_model_t;

typedef struct ai_dnnl_training_progress_t {
    int32_t epoch;
    float train_loss;
    float train_acc;
    float test_loss;
    float test_acc;
} ai_dnnl_training_progress_t;

typedef void(*ai_dnnl_train_callback_t)(ai_dnnl_training_progress_t* progress_info);

uint32_t ai_dnnl_model_create(ai_dnnl_model_t** model, ai_input_dims_t* input_dims, size_t num_layers, ai_dnnl_layer_create_info_t* layer_create_infos, ai_dnnl_loss_kind_t loss_kind);
uint32_t ai_dnnl_model_train(ai_dnnl_model_t* model, size_t train_set_size, float* train_data, uint8_t* train_labels, size_t test_set_size, float* test_data, uint8_t* test_labels, size_t num_epochs, ai_dnnl_train_callback_t callback);
uint32_t ai_dnnl_model_destroy(ai_dnnl_model_t* model);
