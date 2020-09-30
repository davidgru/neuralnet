#pragma once

#include "ai_layer/ai_base_layer.h"
#include "ai_util/ai_loss.h"

typedef struct AI_Net {
    AI_Layer* input_layer;
    AI_Layer** layers;
    size_t num_layers;
    AI_Loss loss;
    size_t input_size;
    size_t output_size;
} AI_Net;

typedef struct AI_TrainingProgress {
    int32_t epoch;
    float train_loss;
    float train_accuracy;
    float test_loss;
    float test_accuracy;
} AI_TrainingProgress;

typedef void (*AI_TrainCallback)(AI_TrainingProgress* progress_info);


void AI_NetInit(AI_Net* net, size_t input_width, size_t input_height, size_t input_channels, size_t batch_size, AI_LayerCreateInfo* create_infos, size_t num_layers);

void AI_NetForward(AI_Net* net, float* input);

void AI_NetTrain(AI_Net* net, float* train_data, float* test_data, uint8_t* train_labels, uint8_t* test_labels, size_t train_dataset_size, size_t test_dataset_size, size_t num_epochs, float learning_rate, size_t batch_size, AI_TrainCallback callback);

void AI_NetDeinit(AI_Net* net);
