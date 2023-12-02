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

#include "config_info.h"
#include "log.h"

#include "dataset.h"
#include "mnist.h"

#include "layer/ai_layer.h"
#include "sequential_model.h"

#include "optimizer/ai_optimizer.h"
#include "optimizer/ai_adam.h"
#include "optimizer/ai_rmsprop.h"
#include "optimizer/ai_sgd.h"


#include "layer_utils.h"


layer_t create_lenet5(const tensor_shape_t* input_shape, float dropout_rate, size_t batch_size)
{
    ai_model_desc_t* desc = NULL;
    layer_t model = NULL;


    /* Allocate resources for the model descriptor. */
    ai_model_desc_create(&desc);

    ai_model_desc_add_convolutional_layer(desc, 6, 5, 1, 0, AI_ConvWeightInitHe, AI_ConvBiasInitZeros);
    ai_model_desc_add_activation_layer(desc, AI_ACTIVATION_FUNCTION_RELU);
    ai_model_desc_add_pooling_layer(desc, 2, 1, 0, AI_POOLING_MAX);
    ai_model_desc_add_dropout_layer(desc, dropout_rate);

    ai_model_desc_add_convolutional_layer(desc, 16, 5, 1, 0, AI_ConvWeightInitHe, AI_ConvBiasInitZeros);
    ai_model_desc_add_activation_layer(desc, AI_ACTIVATION_FUNCTION_RELU);
    ai_model_desc_add_pooling_layer(desc, 2, 1, 0, AI_POOLING_MAX);
    ai_model_desc_add_dropout_layer(desc, dropout_rate);

    ai_model_desc_add_linear_layer(desc, 120, AI_LinearWeightInitHe, AI_LinearBiasInitZeros);
    ai_model_desc_add_activation_layer(desc, AI_ACTIVATION_FUNCTION_RELU);
    ai_model_desc_add_dropout_layer(desc, dropout_rate);

    ai_model_desc_add_linear_layer(desc, 84, AI_LinearWeightInitHe, AI_LinearBiasInitZeros);
    ai_model_desc_add_activation_layer(desc, AI_ACTIVATION_FUNCTION_RELU);
    ai_model_desc_add_dropout_layer(desc, dropout_rate);

    ai_model_desc_add_linear_layer(desc, 10, AI_LinearWeightInitHe, AI_LinearBiasInitZeros);


    /* Print a model overview to stdout. */
    ai_model_desc_dump(desc);

    sequential_model_create_info_t create_info = {
        .desc = desc,
        .max_batch_size = batch_size,
    };
    layer_create(&model, &sequential_model_impl, &create_info, input_shape, batch_size);


    ai_model_desc_destroy(desc);
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

    /* set to location of mnist or fashion_mnist root folder */
    const char* mnist_path = "/home/david/projects/neuralnet/datasets/fashion_mnist";


    /* When training on mnist with this configuration, the model should reach an accuracy of 90%+
        after one epoch and an accuracy of ~98.5% after 10 epochs */
    size_t num_epochs = 10000;
    size_t batch_size = 32;
    AI_LossFunctionEnum loss_type = AI_LOSS_FUNCTION_CROSS_ENTROPY;
    /* use sgd optimizer */
    const optimizer_impl_t* optimizer = &rmsprop_optimizer;
    // sgd_config_t optimizer_config = {
    //     .learning_rate = 2e-2f,
    //     .weight_reg_kind = SGD_WEIGHT_REG_L2,
    //     .weight_reg_strength = 1e-3,
    // };
    rmsprop_config_t optimizer_config = {
        .learning_rate = 1e-3f,
        .gamma = 0.9f,
        .weight_reg_kind = WEIGHT_REG_L2,
        .weight_reg_strength = 1e-3,
    };
    // adam_config_t optimizer_config = {
    //     .learning_rate = 2e-4f,
    //     .gamma1 = 0.9f,
    //     .gamma2 = 0.999f,
    //     .weight_reg_kind = WEIGHT_REG_L2,
    //     .weight_reg_strength = 1e-3f,
    // };
    /* reduce learning rate after 10 epochs without progress on training loss */
    size_t reduce_learning_rate_after = 5;
    float dropout_rate = 0.25f;


    /* Verify the compile time configuration. For example, that avx is used */
    dump_compile_time_config();


    /* Load mnist with a padding of two because lenet expects 32x32 input and the naive
        convolutional layers do not support padding at this time. */
    dataset_t train_set;
    dataset_t test_set;
    mnist_create_info_t mnist_train_info = {
        .path = mnist_path,
        .dataset_kind = MNIST_TRAIN_SET,
        .padding = 2
    };
    mnist_create_info_t mnist_test_info = {
        .path = mnist_path,
        .dataset_kind = MNIST_TEST_SET,
        .padding = 2
    };

    if (dataset_create(&train_set, &mnist_dataset, &mnist_train_info)
        || dataset_create(&test_set, &mnist_dataset, &mnist_test_info)) {
        LOG_ERROR("There was an error loading the mnist dataset\n");
        return 1;
    }
    LOG_INFO("Successfully loaded mnist\n");


    layer_t lenet5 = create_lenet5(dataset_get_shape(train_set), dropout_rate, batch_size);
    LOG_INFO("Created the model. Start training...\n");


    ai_module_train(lenet5, train_set, test_set, num_epochs, batch_size, optimizer,
        &optimizer_config, loss_type, reduce_learning_rate_after, train_callback);


    /* Free resources */
    layer_destroy(lenet5);
    dataset_destroy(train_set);
    dataset_destroy(test_set);

    return 0;
}
