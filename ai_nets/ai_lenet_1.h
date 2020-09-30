#pragma once

#include "../ai_layer/ai_convolutional_layer.h"
#include "../ai_layer/ai_pooling_layer.h"
#include "../ai_layer/ai_linear_layer.h"
#include "../ai_layer/ai_activation_layer.h"
#include "../ai_util/ai_loss.h"
#include "../ai_util/ai_math.h"

#include <string.h>
#include <math.h>

#include <stdio.h>

typedef struct AI_LeNet1 {
    AI_ConvolutionalLayer c1;
    AI_ActivationLayer a1;
    AI_PoolingLayer p1;
    AI_ConvolutionalLayer c2;
    AI_ActivationLayer a2;
    AI_PoolingLayer p2;
    AI_ConvolutionalLayer c3;
    AI_ActivationLayer a3;
    AI_Loss loss;
    size_t input_size;
    size_t output_size;
} AI_LeNet1;


void AI_LeNet1Init(AI_LeNet1* net, float learning_rate)
{
    net->input_size = 28*28;
    net->output_size = 10;

    AI_ConvolutionalLayerInit(&net->c1, 28, 28, 1, 4, 5, learning_rate, 1000.0f, AI_ConvWeightInitXavier, AI_ConvBiasInitZeros);
    AI_ActivationLayerInit(&net->a1, 24*24*4, 1, AI_ACTIVATION_FUNCTION_TANH);
    AI_PoolingLayerInit(&net->p1, 24, 24, 4, 2, AI_POOLING_AVERAGE);
    AI_ConvolutionalLayerInit(&net->c2, 12, 12, 4, 12, 5, learning_rate, 1000.0f, AI_ConvWeightInitXavier, AI_ConvBiasInitZeros);
    AI_ActivationLayerInit(&net->a2, 8*8*12, 1, AI_ACTIVATION_FUNCTION_TANH);
    AI_PoolingLayerInit(&net->p2, 8, 8, 12, 2, AI_POOLING_AVERAGE);
    AI_ConvolutionalLayerInit(&net->c3, 4, 4, 12, 10, 4, learning_rate, 1000.0f, AI_ConvWeightInitXavier, AI_ConvBiasInitZeros);
    AI_ActivationLayerInit(&net->a3, 10, 1, AI_ACTIVATION_FUNCTION_SIGMOID);

    AI_LossInit(&net->loss, 10, 1, AI_LOSS_FUNCTION_MSE);

    AI_LayerLink(&net->c1.hdr,            0, &net->a1.hdr);
    AI_LayerLink(&net->a1.hdr, &net->c1.hdr, &net->p1.hdr);
    AI_LayerLink(&net->p1.hdr, &net->a1.hdr, &net->c2.hdr);
    AI_LayerLink(&net->c2.hdr, &net->p1.hdr, &net->a2.hdr);
    AI_LayerLink(&net->a2.hdr, &net->c2.hdr, &net->p2.hdr);
    AI_LayerLink(&net->p2.hdr, &net->a2.hdr, &net->c3.hdr);
    AI_LayerLink(&net->c3.hdr, &net->p2.hdr, &net->a3.hdr);
    AI_LayerLink(&net->a3.hdr, &net->c3.hdr,            0);    
    AI_LossLink(&net->loss, &net->a3.hdr);
}

void AI_LeNet1Forward(AI_LeNet1* net, float* input)
{
    net->c1.hdr.input = input;
    AI_LayerForward(&net->c1.hdr);
    AI_LayerForward(&net->a1.hdr);
    AI_LayerForward(&net->p1.hdr);
    AI_LayerForward(&net->c2.hdr);
    AI_LayerForward(&net->a2.hdr);
    AI_LayerForward(&net->p2.hdr);
    AI_LayerForward(&net->c3.hdr);
    AI_LayerForward(&net->a3.hdr);
}

typedef struct AI_LeNet1TrainingProgress {
    int32_t epoch;
    float train_loss;
    float train_accuracy;
    float test_loss;
    float test_accuracy;
    float l1_mean;
    float l1_stddev;
    float l2_mean;
    float l2_stddev;
    float l3_mean;
    float l3_stddev;
    float l4_mean;
    float l4_stddev;
} AI_LeNet1TrainingProgress;

typedef void (*AI_LeNet1TrainCallback)(AI_LeNet1TrainingProgress* progress_info);


void AI_LeNet1Train(AI_LeNet1* net, float* train_data, float* test_data, uint8_t* train_labels, uint8_t* test_labels, size_t train_set_size, size_t test_set_size, uint32_t epochs, size_t mini_batch_size, float learning_rate, AI_LeNet1TrainCallback callback)
{
    AI_LeNet1TrainingProgress progress_info;

    memset(&progress_info, 0, sizeof(progress_info));

    float test_loss = 0.0f;
    float test_accuracy = 0.0f;

    // Test
    for (uint32_t j = 0; j < test_set_size; j++) {
        float* input = test_data + j * net->input_size;
        uint8_t* label = test_labels + j;
        AI_LeNet1Forward(net, input);
        test_accuracy += AI_LossAccuracy(&net->loss, label);
        test_loss += AI_LossCompute(&net->loss, label);
    }
    test_accuracy = test_accuracy * 100.0f / test_set_size;
    test_loss /= test_set_size;


    if (callback) {
        progress_info.epoch = -1;
        progress_info.train_loss = 0.0f;
        progress_info.train_accuracy = 0.0f;
        progress_info.test_loss = test_loss;
        progress_info.test_accuracy = test_accuracy;

        callback(&progress_info);
    }

    for (uint32_t i = 0; i < epochs; i++) {

        float train_loss = 0.0f;
        float train_accuracy = 0.0f;

        test_loss = 0.0f;
        test_accuracy = 0.0f;

        // Train one epoch
        for (uint32_t j = 0; j < train_set_size; j++) {
            float* input = train_data + j * net->input_size;
            uint8_t* label = train_labels + j;

            // Forward pass
            AI_LeNet1Forward(net, input);
            
            // Evaluation
            train_accuracy += AI_LossAccuracy(&net->loss, label);
            train_loss += AI_LossCompute(&net->loss, label);

            // Backward pass
            AI_LossBackward(&net->loss, label);
            AI_LayerBackward(&net->a3.hdr);
            AI_LayerBackward(&net->c3.hdr);
            AI_LayerBackward(&net->p2.hdr);
            AI_LayerBackward(&net->a2.hdr);
            AI_LayerBackward(&net->c2.hdr);
            AI_LayerBackward(&net->p1.hdr);
            AI_LayerBackward(&net->a1.hdr);
            AI_LayerBackward(&net->c1.hdr);
        }
        train_loss = train_loss * mini_batch_size / train_set_size;
        train_accuracy = train_accuracy * mini_batch_size * 100.0f / train_set_size;
        
        // Test
        for (uint32_t j = 0; j < test_set_size; j++) {
            float* input = test_data + j * net->input_size;
            uint8_t* label = test_labels + j;
            AI_LeNet1Forward(net, input);
            test_accuracy += AI_LossAccuracy(&net->loss, label);
            test_loss += AI_LossCompute(&net->loss, label);
        }
        test_accuracy = test_accuracy * 100.0f / test_set_size;
        test_loss /= test_set_size;

        if (callback) {
            progress_info.epoch = i;
            progress_info.train_loss = train_loss;
            progress_info.train_accuracy = train_accuracy;
            progress_info.test_loss = test_loss;
            progress_info.test_accuracy = test_accuracy;

            callback(&progress_info);
        }
    }

}


void AI_LeNet1Deinit(AI_LeNet1* net)
{
    AI_LayerDeinit(&net->c1.hdr);
    AI_LayerDeinit(&net->a1.hdr);
    AI_LayerDeinit(&net->p1.hdr);
    AI_LayerDeinit(&net->c2.hdr);
    AI_LayerDeinit(&net->a2.hdr);
    AI_LayerDeinit(&net->p2.hdr);
    AI_LayerDeinit(&net->c3.hdr);
    AI_LayerDeinit(&net->a3.hdr);
    AI_LossDeinit(&net->loss);
}

