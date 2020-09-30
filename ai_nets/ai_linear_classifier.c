
#include "ai_linear_classifier.h"

#include "../ai_layer/ai_activation_layer.h"
#include "../ai_layer/ai_linear_layer.h"

void AI_LinearClassifierInit(AI_LinearClassifier* classifier, size_t input_size, size_t output_size, float learning_rate)
{
    classifier->input_size = input_size;
    classifier->output_size = output_size;

    AI_LinearLayerInit(&classifier->layer, input_size, output_size, 1, learning_rate, AI_LinearWeightInitXavier, AI_LinearBiasInitZeros);
    AI_ActivationLayerInit(&classifier->activation, output_size, 1, AI_ACTIVATION_FUNCTION_SOFTMAX);
    AI_LossInit(&classifier->loss, output_size, 1, AI_LOSS_FUNCTION_MSE);

    AI_LayerLink(&classifier->layer.hdr, 0, &classifier->activation.hdr);
    AI_LayerLink(&classifier->activation.hdr, &classifier->layer.hdr, 0);
    AI_LossLink(&classifier->loss, &classifier->activation.hdr);
}

uint8_t AI_LinearClassifierPredict(AI_LinearClassifier* classifier, float* input)
{
    classifier->layer.hdr.input = input;
    AI_LayerForward(&classifier->layer.hdr);
    AI_LayerForward(&classifier->activation.hdr);
    return AI_Max(classifier->activation.hdr.output, classifier->output_size);
}

void AI_LinearClassifierTrain(AI_LinearClassifier* classifier, float* train_data, float* test_data, uint8_t* train_labels, uint8_t* test_labels, size_t train_set_size, size_t test_set_size, uint32_t epochs, float learning_rate, AI_LinearClassifierTrainCallback callback)
{
    for (uint32_t i = 0; i < epochs; i++) {

        float train_loss = 0.0f;
        float train_accuracy = 0.0f;
        float test_loss = 0.0f;
        float test_accuracy = 0.0f;

        // Train one epoch
        for (uint32_t j = 0; j < train_set_size; j++) {
            float* input = train_data + j * classifier->input_size;
            uint8_t* label = train_labels + j;
            // Forward
            classifier->layer.hdr.input = input;
            AI_LayerForward(&classifier->layer.hdr);
            AI_LayerForward(&classifier->activation.hdr);
            // Evaluate
            train_accuracy += AI_LossAccuracy(&classifier->loss, label);
            train_loss += AI_LossCompute(&classifier->loss, label);
            // Backward
            AI_LossBackward(&classifier->loss, label);
            AI_LayerBackward(&classifier->activation.hdr);
            AI_LayerBackward(&classifier->layer.hdr);
        }

        // Test
        for (uint32_t j = 0; j < test_set_size; j++) {
            float* input = test_data + j * classifier->input_size;
            uint8_t* label = test_labels + j;
            // Forward
            classifier->layer.hdr.input = input;
            AI_LayerForward(&classifier->layer.hdr);
            AI_LayerForward(&classifier->activation.hdr);
            // Evaluate
            test_accuracy += AI_LossAccuracy(&classifier->loss, label);
            test_loss += AI_LossCompute(&classifier->loss, label);
        }

        train_loss /= train_set_size;
        test_loss /= test_set_size;
        train_accuracy = train_accuracy * 100.0f / train_set_size;
        test_accuracy = test_accuracy * 100.0f / test_set_size;

        // Callback
        if (callback)
            callback(i, train_loss, train_accuracy, test_loss, test_accuracy);
        
    }
}

void AI_LinearClassifierDeinit(AI_LinearClassifier* classifier)
{
    AI_LayerDeinit(&classifier->layer.hdr);
    AI_LayerDeinit(&classifier->activation.hdr);
}
