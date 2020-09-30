#pragma once

#include "../ai_layer/ai_linear_layer.h"
#include "../ai_layer/ai_activation_layer.h"
#include "../ai_util/ai_loss.h"

typedef struct AI_2LNet {
    AI_LinearLayer l1;
    AI_LinearLayer l2;
    AI_ActivationLayer a1;
    AI_ActivationLayer a2;
    AI_Loss loss;
    size_t input_size;
} AI_2LNet;

typedef void (*AI_2LNetTrainCallback)(uint32_t epoch, float train_loss, float train_accuracy, float test_loss, float test_accuracy, float l1_mean, float l1_stddev, float l2_mean, float l2_stddev, void* userdata);


uint32_t AI_2LNetInit(AI_2LNet* net, size_t input_size, size_t hidden_size, size_t output_size, size_t mini_batch_size, float learning_rate);

uint8_t AI_2LNetPredict(AI_2LNet* net, float* input);

void AI_2LNetTrain(AI_2LNet* net, float* train_data, float* test_data, uint8_t* train_labels, uint8_t* test_labels, size_t train_set_size, size_t test_set_size, uint32_t epochs, size_t mini_batch_size, float learning_rate, AI_2LNetTrainCallback callback, void* userdata);

void AI_2LNetDeinit(AI_2LNet* net);
