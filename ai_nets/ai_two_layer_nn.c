
#include "ai_two_layer_nn.h"

#include "../ai_layer/ai_base_layer.h"
#include "../ai_util/ai_math.h"

#include <math.h>


static void set_batch_size(AI_2LNet* net, size_t batch_size)
{
    net->l1.hdr.mini_batch_size = batch_size;
    net->a1.hdr.mini_batch_size = batch_size;
    net->l2.hdr.mini_batch_size = batch_size;
    net->a2.hdr.mini_batch_size = batch_size;
    net->loss.mini_batch_size = batch_size;
}


static void nn_forward(AI_2LNet* net, float* input)
{
    net->l1.hdr.input = input;
    AI_LayerForward(&net->l1.hdr);
    AI_LayerForward(&net->a1.hdr);
    AI_LayerForward(&net->l2.hdr);
    AI_LayerForward(&net->a2.hdr);
}


uint32_t AI_2LNetInit(AI_2LNet* net, size_t input_size, size_t hidden_size, size_t output_size, size_t mini_batch_size, float learning_rate)
{
    AI_LinearLayerInit(&net->l1, input_size, hidden_size, mini_batch_size, learning_rate, AI_LinearWeightInitHe, AI_LinearBiasInitZeros);
    AI_LinearLayerInit(&net->l2, hidden_size, output_size,  mini_batch_size, learning_rate, AI_LinearWeightInitXavier, AI_LinearBiasInitZeros);
    AI_ActivationLayerInit(&net->a1, hidden_size, mini_batch_size, AI_ACTIVATION_FUNCTION_LEAKY_RELU);
    AI_ActivationLayerInit(&net->a2, output_size, mini_batch_size, AI_ACTIVATION_FUNCTION_SOFTMAX);

    AI_LossInit(&net->loss, output_size, mini_batch_size, AI_LOSS_FUNCTION_MSE);
    
    AI_LayerLink(&net->l1.hdr,            0, &net->a1.hdr);
    AI_LayerLink(&net->a1.hdr, &net->l1.hdr, &net->l2.hdr);
    AI_LayerLink(&net->l2.hdr, &net->a1.hdr, &net->a2.hdr);
    AI_LayerLink(&net->a2.hdr, &net->l2.hdr, 0);

    AI_LossLink(&net->loss, &net->a2.hdr);

    net->input_size = input_size;

    return 0;
}

uint8_t AI_2LNetPredict(AI_2LNet* net, float* input)
{
    nn_forward(net, input);
    return AI_Max(net->a2.hdr.output, net->a2.hdr.output_width);
}

void AI_2LNetTrain(AI_2LNet* net, float* train_data, float* test_data, uint8_t* train_labels, uint8_t* test_labels, size_t train_set_size, size_t test_set_size, uint32_t epochs, size_t mini_batch_size, float learning_rate, AI_2LNetTrainCallback callback, void* userdata)
{

    float test_loss = 0.0f;
    float test_accuracy = 0.0f;

    // Test
    set_batch_size(net, 1);
    for (uint32_t j = 0; j < test_set_size; j++) {
        nn_forward(net, test_data + j * net->input_size);
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
        set_batch_size(net, mini_batch_size);
        for (uint32_t j = 0; j < train_set_size; j += mini_batch_size) {
            float* train_data_batch = train_data + j * net->input_size;
            uint8_t* train_label_batch = train_labels + j;

            // Forward pass
            nn_forward(net, train_data_batch);

            // Evaluation
            train_accuracy += AI_LossAccuracy(&net->loss, train_label_batch);
            train_loss += AI_LossCompute(&net->loss, train_label_batch);

            // Backward pass
            AI_LossBackward(&net->loss, train_label_batch);
            AI_LayerBackward(&net->a2.hdr);
            AI_LayerBackward(&net->l2.hdr);
            AI_LayerBackward(&net->a1.hdr);
            AI_LayerBackward(&net->l1.hdr);
        }

        // Test
        set_batch_size(net, 1);
        for (uint32_t j = 0; j < test_set_size; j++) {
            nn_forward(net, test_data + j * net->input_size);
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


void AI_2LNetDeinit(AI_2LNet* net)
{
    AI_LayerDeinit(&net->l1.hdr);
    AI_LayerDeinit(&net->l2.hdr);
    AI_LayerDeinit(&net->a1.hdr);
    AI_LayerDeinit(&net->a2.hdr);
    AI_LossDeinit(&net->loss);
}
