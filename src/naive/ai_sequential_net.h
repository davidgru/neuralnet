#pragma once


#include "layer/ai_base_layer.h"
#include "util/ai_loss.h"

#include "ai_model_desc.h"


typedef struct ai_input_dims_t {
    size_t batch_size;
    size_t channels;
    size_t height;
    size_t width;
} ai_input_dims_t;


typedef struct ai_sequential_network {
    AI_Layer* input_layer;
    AI_Layer** layers;
    size_t num_layers;
    size_t input_size;
    size_t output_size;
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
    ai_input_dims_t* input_dims,
    ai_model_desc_t* desc
);


void ai_sequential_network_forward(ai_sequential_network_t* net, float* input);


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
    float learning_rate,
    size_t batch_size,
    AI_LossFunctionEnum loss_type,
    ai_training_callback_t callback
);


void ai_sequential_network_destroy(ai_sequential_network_t* net);
