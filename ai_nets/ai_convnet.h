#pragma once

#include "../ai_layer/ai_convolutional_layer.h"
#include "../ai_layer/ai_linear_layer.h"
#include "../ai_layer/ai_activation_layer.h"
#include "../ai_util/ai_loss.h"
#include "../ai_util/ai_math.h"

#include <string.h>
#include <math.h>

typedef struct AI_ConvNet {
    AI_ConvolutionalLayer c1;
    AI_ActivationLayer a1;
    AI_ConvolutionalLayer c2;
    AI_ActivationLayer a2;
    AI_LinearLayer l3;
    AI_ActivationLayer a3;
    AI_LinearLayer l4;
    AI_ActivationLayer a4;
    AI_Loss loss;
    size_t input_size;
    size_t output_size;
} AI_ConvNet;


void AI_ConvNetInit(AI_ConvNet* net, float learning_rate)
{
    net->input_size = 28*28;
    net->output_size = 10;

    AI_ConvolutionalLayerInit(&net->c1, 28, 28, 1, 4, 5, learning_rate, 1000.0f, AI_ConvWeightInitXavier, AI_ConvBiasInitZeros);
    AI_ConvolutionalLayerInit(&net->c2, 24, 24, 4, 8, 5, learning_rate, 1000.0f, AI_ConvWeightInitXavier, AI_ConvBiasInitZeros);
    
    AI_LinearLayerInit(&net->l3, 20*20*8, 32, 1, learning_rate, 1000.0f, AI_LinearWeightInitXavier, AI_LinearBiasInitZeros);
    AI_LinearLayerInit(&net->l4, 32, 10, 1, learning_rate, 1000.0f, AI_LinearWeightInitXavier, AI_LinearBiasInitZeros);

    AI_ActivationLayerInit(&net->a1, 24*24*4, 1, AI_ACTIVATION_FUNCTION_SIGMOID);
    AI_ActivationLayerInit(&net->a2, 20*20*8, 1, AI_ACTIVATION_FUNCTION_SIGMOID);
    AI_ActivationLayerInit(&net->a3, 32, 1, AI_ACTIVATION_FUNCTION_SIGMOID);
    AI_ActivationLayerInit(&net->a4, 10, 1, AI_ACTIVATION_FUNCTION_SOFTMAX);

    AI_LossInit(&net->loss, 10, 1, AI_LOSS_FUNCTION_MSE);

    AI_LayerLink(&net->c1.hdr,            0, &net->a1.hdr);
    AI_LayerLink(&net->a1.hdr, &net->c1.hdr, &net->c2.hdr);
    AI_LayerLink(&net->c2.hdr, &net->a1.hdr, &net->a2.hdr);
    AI_LayerLink(&net->a2.hdr, &net->c2.hdr, &net->l3.hdr);
    AI_LayerLink(&net->l3.hdr, &net->a2.hdr, &net->a3.hdr);
    AI_LayerLink(&net->a3.hdr, &net->l3.hdr, &net->l4.hdr);
    AI_LayerLink(&net->l4.hdr, &net->a3.hdr, &net->a4.hdr);
    AI_LayerLink(&net->a4.hdr, &net->l4.hdr, 0);
    AI_LossLink(&net->loss, &net->a4.hdr);
}

void AI_ConvNetForward(AI_ConvNet* net, float* input)
{
    net->c1.hdr.input = input;
    AI_LayerForward(&net->c1.hdr);
    AI_LayerForward(&net->a1.hdr);
    AI_LayerForward(&net->c2.hdr);
    AI_LayerForward(&net->a2.hdr);
    AI_LayerForward(&net->l3.hdr);
    AI_LayerForward(&net->a3.hdr);
    AI_LayerForward(&net->l4.hdr);
    AI_LayerForward(&net->a4.hdr);
}

uint32_t AI_ConvNetPredict(AI_ConvNet* net, float* input)
{
    AI_ConvNetForward(net, input);
    return AI_Max(net->a4.hdr.output, 10);
}


typedef struct AI_ConvNetTrainingProgress {
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
} AI_ConvNetTrainingProgress;

typedef void (*AI_ConvNetTrainCallback)(AI_ConvNetTrainingProgress* progress_info);


void AI_ConvNetTrain(AI_ConvNet* net, float* train_data, float* test_data, uint8_t* train_labels, uint8_t* test_labels, size_t train_set_size, size_t test_set_size, uint32_t epochs, size_t mini_batch_size, float learning_rate, AI_ConvNetTrainCallback callback)
{
    AI_ConvNetTrainingProgress progress_info;

    memset(&progress_info, 0, sizeof(progress_info));

    float test_loss = 0.0f;
    float test_accuracy = 0.0f;

    // Test
    for (uint32_t j = 0; j < test_set_size; j++) {
        float* input = test_data + j * net->input_size;
        uint8_t* label = test_labels + j;
        AI_ConvNetForward(net, input);
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

        progress_info.l1_mean = AI_Mean(net->c1.w, net->c1.filter_width * net->c1.filter_width * net->c1.hdr.input_channels * net->c1.hdr.output_channels);
        progress_info.l1_stddev = AI_Stddev(net->c1.w, net->c1.filter_width * net->c1.filter_width * net->c1.hdr.input_channels * net->c1.hdr.output_channels);
        progress_info.l2_mean = AI_Mean(net->c2.w, net->c2.filter_width * net->c2.filter_width * net->c2.hdr.input_channels * net->c2.hdr.output_channels);
        progress_info.l2_stddev = AI_Stddev(net->c2.w, net->c2.filter_width * net->c2.filter_width * net->c2.hdr.input_channels * net->c2.hdr.output_channels);
        progress_info.l3_mean = AI_FCLayerMeanWeights(&net->l3);
        progress_info.l3_stddev = AI_FCLayerStddevWeights(&net->l3);
        progress_info.l4_mean = AI_FCLayerMeanWeights(&net->l4);
        progress_info.l4_stddev = AI_FCLayerStddevWeights(&net->l4);

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
            AI_ConvNetForward(net, input);
            
            // Evaluation
            train_accuracy += AI_LossAccuracy(&net->loss, label);
            train_loss += AI_LossCompute(&net->loss, label);

            // Backward pass
            AI_LossBackward(&net->loss, label);
            AI_LayerBackward(&net->a4.hdr);
            AI_LayerBackward(&net->l4.hdr);
            AI_LayerBackward(&net->a3.hdr);
            AI_LayerBackward(&net->l3.hdr);
            AI_LayerBackward(&net->a2.hdr);
            AI_LayerBackward(&net->c2.hdr);
            AI_LayerBackward(&net->a1.hdr);
            AI_LayerBackward(&net->c1.hdr);
        }
        train_loss = train_loss * mini_batch_size / train_set_size;
        train_accuracy = train_accuracy * mini_batch_size * 100.0f / train_set_size;
        
        // Test
        for (uint32_t j = 0; j < test_set_size; j++) {
            float* input = test_data + j * net->input_size;
            uint8_t* label = test_labels + j;
            AI_ConvNetForward(net, input);
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

            progress_info.l1_mean = AI_Mean(net->c1.w, net->c1.filter_width * net->c1.filter_width * net->c1.hdr.input_channels * net->c1.hdr.output_channels);
            progress_info.l1_stddev = AI_Stddev(net->c1.w, net->c1.filter_width * net->c1.filter_width * net->c1.hdr.input_channels * net->c1.hdr.output_channels);
            progress_info.l2_mean = AI_Mean(net->c2.w, net->c2.filter_width * net->c2.filter_width * net->c2.hdr.input_channels * net->c2.hdr.output_channels);
            progress_info.l2_stddev = AI_Stddev(net->c2.w, net->c2.filter_width * net->c2.filter_width * net->c2.hdr.input_channels * net->c2.hdr.output_channels);
            progress_info.l3_mean = AI_FCLayerMeanWeights(&net->l3);
            progress_info.l3_stddev = AI_FCLayerStddevWeights(&net->l3);
            progress_info.l4_mean = AI_FCLayerMeanWeights(&net->l4);
            progress_info.l4_stddev = AI_FCLayerStddevWeights(&net->l4);

            callback(&progress_info);
        }
    }

}


void AI_ConvNetDeinit(AI_ConvNet* net)
{

}

