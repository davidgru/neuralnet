#pragma once

#pragma once

#include "../ai_layer/ai_linear_layer.h"
#include "../ai_layer/ai_activation_layer.h"
#include "../ai_util/ai_loss.h"

typedef struct AI_DeepNet {
    AI_LinearLayer l1;
    AI_LinearLayer l2;
    AI_LinearLayer l3;
    AI_LinearLayer l4;
    
    AI_ActivationLayer a1;
    AI_ActivationLayer a2;
    AI_ActivationLayer a3;
    AI_ActivationLayer a4;
    AI_Loss loss;
    size_t input_size;
    size_t output_size;
} AI_DeepNet;

typedef void (*AI_DeepNetTrainCallback)(uint32_t epoch, float train_loss, float train_accuracy, float test_loss, float test_accuracy, float l1_mean, float l1_stddev, float l2_mean, float l2_stddev, void* userdata);


uint32_t AI_DeepNetInit(AI_DeepNet* net, size_t mini_batch_size, float learning_rate)
{
    AI_LinearLayerInit(&net->l1, 28*28, 800, mini_batch_size, learning_rate, 1000.0f, AI_LinearWeightInitXavier, AI_LinearBiasInitZeros);
    AI_ActivationLayerInit(&net->a1, 800, mini_batch_size, AI_ACTIVATION_FUNCTION_SIGMOID);
    
    AI_LinearLayerInit(&net->l2, 800, 400, mini_batch_size, learning_rate, 1000.0f, AI_LinearWeightInitXavier, AI_LinearBiasInitZeros);
    AI_ActivationLayerInit(&net->a2, 400, mini_batch_size, AI_ACTIVATION_FUNCTION_SIGMOID);
    
    AI_LinearLayerInit(&net->l3, 400, 100, mini_batch_size, learning_rate, 1000.0f, AI_LinearWeightInitXavier, AI_LinearBiasInitZeros);
    AI_ActivationLayerInit(&net->a3, 100, mini_batch_size, AI_ACTIVATION_FUNCTION_SIGMOID);

    AI_LinearLayerInit(&net->l4, 100, 10, mini_batch_size, learning_rate, 1000.0f, AI_LinearWeightInitXavier, AI_LinearBiasInitZeros);    
    AI_ActivationLayerInit(&net->a4,  10, mini_batch_size, AI_ACTIVATION_FUNCTION_SOFTMAX);

    AI_LossInit(&net->loss, 10, mini_batch_size, AI_LOSS_FUNCTION_MSE);

    AI_LayerLink(&net->l1.hdr, 0, &net->a1.hdr);
    AI_LayerLink(&net->a1.hdr, &net->l1.hdr, &net->l2.hdr);
    AI_LayerLink(&net->l2.hdr, &net->a1.hdr, &net->a2.hdr);
    AI_LayerLink(&net->a2.hdr, &net->l2.hdr, &net->l3.hdr);
    AI_LayerLink(&net->l3.hdr, &net->a2.hdr, &net->a3.hdr);
    AI_LayerLink(&net->a3.hdr, &net->l3.hdr, &net->l4.hdr);
    AI_LayerLink(&net->l4.hdr, &net->a3.hdr, &net->a4.hdr);
    AI_LayerLink(&net->a4.hdr, &net->l4.hdr, 0);
    AI_LossLink(&net->loss, &net->a4.hdr);

    net->input_size = 28*28;
    net->output_size = 10;
}

uint8_t AI_DeepNetForward(AI_DeepNet* net, float* input)
{
    net->l1.hdr.input = input;
    AI_LayerForward(&net->l1.hdr);
    AI_LayerForward(&net->a1.hdr);
    AI_LayerForward(&net->l2.hdr);
    AI_LayerForward(&net->a2.hdr);
    AI_LayerForward(&net->l3.hdr);
    AI_LayerForward(&net->a3.hdr);
    AI_LayerForward(&net->l4.hdr);
    AI_LayerForward(&net->a4.hdr);
}

void AI_DeepNetSetBatchSize(AI_DeepNet* net, size_t mini_batch_size)
{
    net->l1.hdr.mini_batch_size = mini_batch_size;
    net->a1.hdr.mini_batch_size = mini_batch_size;
    net->l2.hdr.mini_batch_size = mini_batch_size;
    net->a2.hdr.mini_batch_size = mini_batch_size;
    net->l3.hdr.mini_batch_size = mini_batch_size;
    net->a3.hdr.mini_batch_size = mini_batch_size;
    net->l4.hdr.mini_batch_size = mini_batch_size;
    net->a4.hdr.mini_batch_size = mini_batch_size;
    net->loss.mini_batch_size = mini_batch_size;
}

void AI_DeepNetTrain(AI_DeepNet* net, float* train_data, float* test_data, uint8_t* train_labels, uint8_t* test_labels, size_t train_set_size, size_t test_set_size, uint32_t epochs, size_t mini_batch_size, float learning_rate, AI_DeepNetTrainCallback callback, void* userdata)
{
    float test_loss = 0.0f;
    float test_accuracy = 0.0f;

    // Test
    AI_DeepNetSetBatchSize(net, 1);
    for (uint32_t j = 0; j < test_set_size; j++) {
        AI_DeepNetForward(net, test_data + j * net->input_size);
        test_accuracy += AI_LossAccuracy(&net->loss, test_labels + j);
        test_loss += AI_LossCompute(&net->loss, test_labels + j);
    }
    test_accuracy = test_accuracy * 100.0f / test_set_size;
    test_loss /= test_set_size;

    if (callback)
        callback(-1, 0, 0, test_loss, test_accuracy, AI_FCLayerMeanWeights(&net->l1), AI_FCLayerStddevWeights(&net->l1), AI_FCLayerMeanWeights(&net->l2), AI_FCLayerStddevWeights(&net->l2), userdata);
    
    for (uint32_t i = 0; i < epochs; i++) {

        float train_loss = 0.0f;
        float train_accuracy = 0.0f;

        float l1_mean = 0.0f;
        float l1_stddev = 0.0f;
        float l2_mean = 0.0f;
        float l2_stddev = 0.0f;

        // Train one epoch
        AI_DeepNetSetBatchSize(net, mini_batch_size);
        for (uint32_t j = 0; j < train_set_size; j += mini_batch_size) {
            float* train_data_batch = train_data + j * net->input_size;
            uint8_t* train_label_batch = train_labels + j;

            // Forward pass
            AI_DeepNetForward(net, train_data_batch);

            // Evaluation
            train_accuracy += AI_LossAccuracy(&net->loss, train_label_batch);
            train_loss += AI_LossCompute(&net->loss, train_label_batch);

            // Backward pass
            AI_LossBackward(&net->loss, train_label_batch);
            AI_LayerBackward(&net->a4.hdr);
            AI_LayerBackward(&net->l4.hdr);
            AI_LayerBackward(&net->a3.hdr);
            AI_LayerBackward(&net->l3.hdr);
            AI_LayerBackward(&net->a2.hdr);
            AI_LayerBackward(&net->l2.hdr);
            AI_LayerBackward(&net->a1.hdr);
            AI_LayerBackward(&net->l1.hdr);
        }

        // Test
        AI_DeepNetSetBatchSize(net, 1);
        for (uint32_t j = 0; j < test_set_size; j++) {
            AI_DeepNetForward(net, test_data + j * net->input_size);
            test_accuracy += AI_LossAccuracy(&net->loss, test_labels + j);
            test_loss += AI_LossCompute(&net->loss, test_labels + j);
        }
    
        train_loss = train_loss / train_set_size;
        test_loss = test_loss / test_set_size;
        train_accuracy = train_accuracy * 100.0f / train_set_size;
        test_accuracy = test_accuracy * 100.0f / test_set_size;


        l1_mean = AI_FCLayerMeanWeights(&net->l1);
        l1_stddev = AI_FCLayerStddevWeights(&net->l1);
        l2_mean = AI_FCLayerMeanWeights(&net->l2);
        l2_stddev = AI_FCLayerStddevWeights(&net->l2);

        if (callback)
            callback(i, train_loss, train_accuracy, test_loss, test_accuracy, l1_mean, l1_stddev, l2_mean, l2_stddev, userdata);
    }
}

void AI_DeepNetDeinit(AI_DeepNet* net)
{
    AI_LayerDeinit(&net->l1.hdr);
    AI_LayerDeinit(&net->a1.hdr);
    AI_LayerDeinit(&net->l2.hdr);
    AI_LayerDeinit(&net->a2.hdr);
    AI_LayerDeinit(&net->l3.hdr);
    AI_LayerDeinit(&net->a3.hdr);
    AI_LayerDeinit(&net->l4.hdr);
    AI_LayerDeinit(&net->a4.hdr);
    AI_LossDeinit(&net->loss);
}
