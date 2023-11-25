#pragma once


#include "tensor.h"

#include "ai_model_desc.h"
#include "layer/ai_base_layer.h"
#include "util/ai_loss.h"

#include "layer/ai_layer.h"
#include "optimizer/ai_optimizer.h"


typedef struct ai_sequential_network {
    layer_t* layers;
    size_t num_layers;
    tensor_shape_t input_shape;
    tensor_shape_t output_shape;
} ai_sequential_network_t;


typedef struct ai_training_info {
    int32_t epoch;
    float train_loss;
    float train_accuracy;
    float test_loss;
    float test_accuracy;
} ai_training_info_t;


typedef void (*ai_training_callback_t)(ai_training_info_t * progress_info);



void ai_sequential_network_create(
    ai_sequential_network_t** net,
    tensor_shape_t* input_shape,
    size_t max_batch_size,
    ai_model_desc_t* desc
);


void ai_sequential_network_forward(
    ai_sequential_network_t* net,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t** out_output
);


void ai_sequential_network_test(
    ai_sequential_network_t* net,
    float* test_data,
    uint8_t* test_labels,
    size_t test_set_size,
    AI_Loss* loss,
    float* out_accuracy,
    float* out_loss
);


void ai_sequential_network_train(
    ai_sequential_network_t* net,
    float* train_data,
    float* test_data,
    uint8_t* train_labels,
    uint8_t* test_labels,
    size_t train_dataset_size,
    size_t test_dataset_size,
    size_t num_epochs,
    size_t batch_size,
    const optimizer_impl_t* optimizer_impl,
    const optimizer_config_t* optimizer_config,
    AI_LossFunctionEnum loss_type,
    ai_training_callback_t callback
);


void ai_sequential_network_destroy(ai_sequential_network_t* net);
