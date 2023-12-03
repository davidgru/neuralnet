/**
 * @file lenet5_mnist.c
 * @brief Train LeNet-5 on the MNIST dataset
 * 
 * This example illustrates how this library can be used to implement the LeNet-5 architecture
 * and trains it on the MNIST dataset.
 */


#include <stdio.h>
#include <inttypes.h>

#include "core/layer.h"
#include "core/loss.h"
#include "core/optimizer.h"

#include "optimizer/sgd.h"

#include "sequential/model_desc.h"
#include "sequential/sequential_model.h"

#include "dataset.h"
#include "mnist.h"

#include "training_utils.h"

#include "config_info.h"
#include "log.h"
#include "tensor.h"


layer_t create_lenet5(const tensor_shape_t* input_shape, size_t batch_size)
{
    model_desc_t* desc = NULL;
    layer_t model = NULL;


    /* Some default configurations */
    activation_layer_create_info_t act_config = {
        .activation_function = ACTIVATION_FUNCTION_TANH,
    };

    pooling_layer_create_info_t pool_config = {
        .kernel_width = 2,
        .pooling_operation = POOLING_AVERAGE,
    };
    linear_layer_create_info_t linear_default_config = {
        .weight_init = linear_weight_init_xavier,
        .bias_init = linear_bias_init_zeros,
    };


    /* Allocate resources for the model descriptor. */
    model_desc_create(&desc);


    convolutional_layer_create_info_t conv1_config = conv_default_config;
    conv1_config.output_channels = 6;
    conv1_config.filter_height = 5;
    conv1_config.filter_width = 5;
    conv1_config.padding_y = 2;
    conv1_config.padding_x = 2;
    model_desc_add_layer(desc, &convolutional_layer_impl, &conv1_config);
    model_desc_add_layer(desc, &activation_layer_impl, &act_config);
    model_desc_add_layer(desc, &pooling_layer_impl, &pool_config);

    convolutional_layer_create_info_t conv2_config = conv_default_config;
    conv2_config.output_channels = 16;
    conv2_config.filter_height = 5;
    conv2_config.filter_width = 5;
    model_desc_add_layer(desc, &convolutional_layer_impl, &conv2_config);
    model_desc_add_layer(desc, &activation_layer_impl, &act_config);
    model_desc_add_layer(desc, &pooling_layer_impl, &pool_config);

    linear_layer_create_info_t linear1_config = linear_default_config;
    linear1_config.output_size = 120;
    model_desc_add_layer(desc, &linear_layer_impl, &linear1_config);
    model_desc_add_layer(desc, &activation_layer_impl, &act_config);

    linear_layer_create_info_t linear2_config = linear_default_config;
    linear2_config.output_size = 84;
    model_desc_add_layer(desc, &linear_layer_impl, &linear2_config);
    model_desc_add_layer(desc, &activation_layer_impl, &act_config);

    linear_layer_create_info_t linear3_config = linear_default_config;
    linear3_config.output_size = 10;
    model_desc_add_layer(desc, &linear_layer_impl, &linear3_config);


    /* Print a model overview to stdout. */
    model_desc_dump(desc);


    sequential_model_create_info_t create_info = {
        .desc = desc,
        .max_batch_size = batch_size,
    };
    layer_create(&model, &sequential_model_impl, &create_info, input_shape, batch_size);


    model_desc_destroy(desc);
    return model;
}


dataset_t load_mnist(const char* path, mnist_dataset_kind_t dataset_kind)
{
    dataset_t dataset = NULL;
    mnist_create_info_t mnist_train_info = {
        .path = path,
        .dataset_kind = dataset_kind,
        .padding = 0
    };
    dataset_create(&dataset, &mnist_dataset, &mnist_train_info);
    return dataset;
}


void train_callback(training_info_t* p)
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
    const char* mnist_path = "/home/david/projects/neuralnet/datasets/mnist";


    /* When training on mnist with this configuration, the model should reach an accuracy of 90%+
        after one epoch and an accuracy of ~98.5% after 10 epochs */
    size_t num_epochs = 10;
    size_t batch_size = 32;
    LossFunctionEnum loss_type = LOSS_FUNCTION_CROSS_ENTROPY;
    /* use sgd optimizer */
    const optimizer_impl_t* optimizer = &sgd_optimizer; 
    sgd_config_t optimizer_config = {
        .learning_rate = 1e-2f,
        .weight_reg_kind = WEIGHT_REG_NONE,
        .weight_reg_strength = 0.0f
    };


    /* Verify the compile time configuration. For example, that avx is used */
    dump_compile_time_config();


    dataset_t train_set = load_mnist(mnist_path, MNIST_TRAIN_SET);
    dataset_t test_set = load_mnist(mnist_path, MNIST_TEST_SET);
    
    if (train_set == NULL || test_set == NULL) {
        LOG_ERROR("There was an error loading the mnist dataset\n");
        return 1;
    }
    LOG_INFO("Successfully loaded mnist\n");


    layer_t lenet5 = create_lenet5(dataset_get_shape(train_set), batch_size);
    LOG_INFO("Created the model. Start training...\n");


    module_train(lenet5, train_set, test_set, num_epochs, batch_size, optimizer, &optimizer_config,
        loss_type, 0, train_callback);


    /* Free resources */
    layer_destroy(lenet5);
    dataset_destroy(train_set);
    dataset_destroy(test_set);

    return 0;
}
