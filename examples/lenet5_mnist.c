/**
 * @file lenet5_mnist.c
 * @brief Train LeNet-5 on the MNIST dataset
 * 
 * This example illustrates how this library can be used to implement the LeNet-5 architecture
 * and trains it on the MNIST dataset.
 */


#include <stdio.h>
#include <inttypes.h>

#include "tensor.h"
#include "tensor_impl.h"

#include "ai_model_desc.h"
#include "ai_sequential_net.h"
#include "ai_mnist.h"

#include "config_info.h"
#include "log.h"


ai_sequential_network_t* create_lenet5(float learning_rate, tensor_shape_t* input_shape, size_t batch_size)
{
    ai_model_desc_t* desc = NULL;
    ai_sequential_network_t* model = NULL;


    /* Allocate resources for the model descriptor. */
    ai_model_desc_create(&desc);

    ai_model_desc_add_convolutional_layer(desc, learning_rate, 6, 5, 1, 0, AI_ConvWeightInitXavier, AI_ConvBiasInitZeros);
    ai_model_desc_add_activation_layer(desc, AI_ACTIVATION_FUNCTION_TANH);
    ai_model_desc_add_pooling_layer(desc, 2, 1, 0, AI_POOLING_AVERAGE);

    ai_model_desc_add_convolutional_layer(desc, learning_rate, 16, 5, 1, 0, AI_ConvWeightInitXavier, AI_ConvBiasInitZeros);
    ai_model_desc_add_activation_layer(desc, AI_ACTIVATION_FUNCTION_TANH);
    ai_model_desc_add_pooling_layer(desc, 2, 1, 0, AI_POOLING_AVERAGE);

    ai_model_desc_add_linear_layer(desc, learning_rate, 120, AI_LinearWeightInitXavier, AI_LinearBiasInitZeros);
    ai_model_desc_add_activation_layer(desc, AI_ACTIVATION_FUNCTION_TANH);

    ai_model_desc_add_linear_layer(desc, learning_rate, 84, AI_LinearWeightInitXavier, AI_LinearBiasInitZeros);
    ai_model_desc_add_activation_layer(desc, AI_ACTIVATION_FUNCTION_TANH);

    ai_model_desc_add_linear_layer(desc, learning_rate, 10, AI_LinearWeightInitXavier, AI_LinearBiasInitZeros);
    ai_model_desc_add_activation_layer(desc, AI_ACTIVATION_FUNCTION_SIGMOID);


    /* Print a model overview to stdout. */
    ai_model_desc_dump(desc);


    ai_sequential_network_create(&model, input_shape, batch_size, desc);

    return model;
}



void train_callback(ai_training_info_t* p)
{
    printf("Epoch %" PRIi32 " | Train loss %f | Train accuracy %5.3f%% | Test loss %f "
        "| Test accuracy %5.3f%%\n",
        p->epoch,
        p->train_loss,
        p->train_accuracy * 100.0f,
        p->test_loss,
        p->test_accuracy * 100.0f
    );
}


int main()
{
    AI_MnistDataset mnist;
    ai_sequential_network_t* lenet5;


    /* set to location of mnist or fashion_mnist root folder */
    const char* mnist_path = ;


    /* When training on mnist with this configuration, the model should reach an accuracy of 90%+
        after one epoch and an accuracy of ~98.5% after 10 epochs */
    size_t num_epochs = 10;
    size_t batch_size = 1; /* Only batch size of 1 supported at the moment */
    float learning_rate = 0.01f;
    AI_LossFunctionEnum loss_type = AI_LOSS_FUNCTION_MSE;


    /* Verify the compile time configuration. For example, that avx is used */
    dump_compile_time_config();


    /* Load mnist with a padding of two because lenet expects 32x32 input and the naive
        convolutional layers do not support padding at this time. */
    if (AI_MnistDatasetLoad(&mnist, mnist_path, 2) != 0) {
        LOG_ERROR("There was an error loading the mnist dataset\n");
        return 1;
    }
    LOG_INFO("Successfully loaded mnist\n");


    tensor_shape_t input_shape = {
        .dims[TENSOR_BATCH_DIM] = batch_size,
        .dims[TENSOR_CHANNEL_DIM] = 1,
        .dims[TENSOR_HEIGHT_DIM] = mnist.image_height,
        .dims[TENSOR_WIDTH_DIM] = mnist.image_width
    };
    lenet5 = create_lenet5(learning_rate, &input_shape, batch_size);
    LOG_INFO("Created the model. Start training...\n");


    ai_sequential_network_train(lenet5, mnist.train_images, mnist.test_images, mnist.train_labels,
        mnist.test_labels, mnist.num_train_images, mnist.num_test_images, num_epochs, learning_rate,
        batch_size, loss_type, train_callback);


    /* Free resources */
    ai_sequential_network_destroy(lenet5);
    AI_MnistDatasetFree(&mnist);

    return 0;
}
