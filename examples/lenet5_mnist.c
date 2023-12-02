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


#include "dataset.h"
#include "mnist.h"

#include "config_info.h"
#include "log.h"

#include "optimizer/ai_optimizer.h"
#include "optimizer/ai_sgd.h"


#include "core/ai_layer.h"
#include "ai_model_desc.h"
#include "sequential_model.h"
#include "layer_utils.h"


layer_t create_lenet5(const tensor_shape_t* input_shape, size_t batch_size)
{
    ai_model_desc_t* desc = NULL;
    layer_t model = NULL;


    /* Allocate resources for the model descriptor. */
    ai_model_desc_create(&desc);

    ai_model_desc_add_convolutional_layer(desc, 6, 5, 1, 0, AI_ConvWeightInitXavier, AI_ConvBiasInitZeros);
    ai_model_desc_add_activation_layer(desc, AI_ACTIVATION_FUNCTION_TANH);
    ai_model_desc_add_pooling_layer(desc, 2, 1, 0, AI_POOLING_AVERAGE);

    ai_model_desc_add_convolutional_layer(desc, 16, 5, 1, 0, AI_ConvWeightInitXavier, AI_ConvBiasInitZeros);
    ai_model_desc_add_activation_layer(desc, AI_ACTIVATION_FUNCTION_TANH);
    ai_model_desc_add_pooling_layer(desc, 2, 1, 0, AI_POOLING_AVERAGE);

    ai_model_desc_add_linear_layer(desc, 120, AI_LinearWeightInitXavier, AI_LinearBiasInitZeros);
    ai_model_desc_add_activation_layer(desc, AI_ACTIVATION_FUNCTION_TANH);

    ai_model_desc_add_linear_layer(desc, 84, AI_LinearWeightInitXavier, AI_LinearBiasInitZeros);
    ai_model_desc_add_activation_layer(desc, AI_ACTIVATION_FUNCTION_TANH);

    ai_model_desc_add_linear_layer(desc, 10, AI_LinearWeightInitXavier, AI_LinearBiasInitZeros);
    ai_model_desc_add_activation_layer(desc, AI_ACTIVATION_FUNCTION_SIGMOID);


    /* Print a model overview to stdout. */
    ai_model_desc_dump(desc);


    sequential_model_create_info_t create_info = {
        .desc = desc,
        .max_batch_size = batch_size,
    };
    layer_create(&model, &sequential_model_info, &create_info, input_shape, batch_size);

    return model;
}


dataset_t load_mnist(const char* path, mnist_dataset_kind_t dataset_kind, size_t padding)
{
    dataset_t dataset = NULL;
    mnist_create_info_t mnist_train_info = {
        .path = path,
        .dataset_kind = MNIST_TRAIN_SET,
        .padding = padding
    };
    dataset_create(&dataset, &mnist_dataset, &mnist_train_info);
    return dataset;
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
    layer_t lenet5;

    /* set to location of mnist or fashion_mnist root folder */
    const char* mnist_path = "/home/david/projects/neuralnet/datasets/mnist";


    /* When training on mnist with this configuration, the model should reach an accuracy of 90%+
        after one epoch and an accuracy of ~98.5% after 10 epochs */
    size_t num_epochs = 10;
    size_t batch_size = 32;
    AI_LossFunctionEnum loss_type = AI_LOSS_FUNCTION_MSE;
    /* use sgd optimizer */
    const optimizer_impl_t* optimizer = &sgd_optimizer; 
    sgd_config_t optimizer_config = {
        .learning_rate = 1e-1f,
        .weight_reg_kind = WEIGHT_REG_NONE,
        .weight_reg_strength = 0.0f
    };


    /* Verify the compile time configuration. For example, that avx is used */
    dump_compile_time_config();


    /* Load mnist with a padding of two because lenet expects 32x32 input and the naive
        convolutional layers do not support padding at this time. */
    dataset_t train_set = load_mnist(mnist_path, MNIST_TRAIN_SET, 2);
    dataset_t test_set = load_mnist(mnist_path, MNIST_TEST_SET, 2);
    
    if (train_set == NULL || test_set == NULL) {
        LOG_ERROR("There was an error loading the mnist dataset\n");
        return 1;
    }
    LOG_INFO("Successfully loaded mnist\n");


    lenet5 = create_lenet5(dataset_get_shape(train_set), batch_size);
    LOG_INFO("Created the model. Start training...\n");


    ai_module_train(lenet5, train_set, test_set, num_epochs, batch_size, optimizer, &optimizer_config, loss_type, train_callback);


    /* Free resources */
    dataset_destroy(train_set);
    dataset_destroy(test_set);

    return 0;
}
