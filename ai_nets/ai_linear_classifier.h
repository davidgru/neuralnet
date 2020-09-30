
#include "../ai_layer/ai_linear_layer.h"
#include "../ai_layer/ai_activation_layer.h"
#include "../ai_util/ai_loss.h"

/*

 * Solving the mnist dataset with a linear classifier.
 * Test accuracy: 92,5%

#include <stdio.h>

#include "ai_datasets/ai_mnist.h"
#include "ai_nets/ai_linear_classifier.h"

void train_callback(uint32_t epoch, float train_loss, float train_accuracy, float test_loss, float test_accuracy)
{
    printf("Epoch %3d | Train loss %f | Train accuracy %.2f%% | Test loss %f | Test accuracy %.2f%% | \n", epoch, train_loss, train_accuracy, test_loss, test_accuracy);
}

int main()
{

    uint32_t epochs = 10000;
    float learning_rate = 0.0001f;

    AI_MnistDataset dataset;
    AI_LinearClassifier classifier;
    
    AI_MnistDatasetLoad(&dataset, "D:/dev/tools/datasets/mnist");
    AI_LinearClassifierInit(&classifier, 28*28, 10, learning_rate);

    AI_LinearClassifierTrain(&classifier, dataset.train_images, dataset.test_images, dataset.train_labels, dataset.test_labels, dataset.num_train_images, dataset.num_test_images, epochs, learning_rate, train_callback);


    AI_LinearClassifierDeinit(&classifier);
    AI_MnistDatasetFree(&dataset);
}

*/

typedef struct AI_LinearClassifier {
    size_t input_size;
    size_t output_size;
    AI_LinearLayer layer;
    AI_ActivationLayer activation;
    AI_Loss loss;
} AI_LinearClassifier;

// Callback is invoked after every training epoch
typedef void (*AI_LinearClassifierTrainCallback)(uint32_t epoch, float train_loss, float train_accuracy, float test_loss, float test_accuracy);

// Init a Linear Classifier
void AI_LinearClassifierInit(AI_LinearClassifier* classifier, size_t input_size, size_t output_size, float learning_rate);

// Forward pass through the Classifier
uint8_t AI_LinearClassifierPredict(AI_LinearClassifier* classifier, float* input);

// Train a Classifier on a training set
void AI_LinearClassifierTrain(AI_LinearClassifier* classifier, float* train_data, float* test_data, uint8_t* train_labels, uint8_t* test_labels, size_t training_set_size, size_t test_set_size, uint32_t epochs, float learning_rate, AI_LinearClassifierTrainCallback callback);

// Deinit a Classifier
void AI_LinearClassifierDeinit(AI_LinearClassifier* classifier);
