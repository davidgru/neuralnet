/**
 * @file lenet5_mnist.c
 * @brief Train LeNet-5 on the MNIST dataset
 * 
 * This example illustrates how this library can be used to implement the LeNet-5 architecture
 * and trains it on the MNIST dataset.
 */

#include <inttypes.h>
#include <math.h>

#include "neuralnet.h"
#include "sequential/model_desc.h"
#include "sequential/sequential_model.h"
#include "dataset/mnist.h"
#include "optimizer/sgd.h"

/* path to MNIST or Fashion MNIST dataset */
const char* mnist_path = "/home/david/projects/neuralnet/datasets/fashion_mnist";

/* When training on mnist with this configuration, the model should reach an accuracy of 90%+
    after one epoch and an accuracy of ~98.5% after 10 epochs */
static const size_t num_epochs = 1000;
static const size_t batch_size = 32;
static const sgd_config_t optimizer_config = {
    .learning_rate = 2e-2f,
    .weight_reg_kind = WEIGHT_REG_NONE,
};


layer_t create_lenet5(const tensor_shape_t* input_shape, size_t batch_size)
{
    model_desc_t* desc = NULL;
    layer_t model = NULL;

    /* Some default configurations */
    const activation_layer_create_info_t act_config = {
        .activation_function = ACTIVATION_FUNCTION_TANH,
    };
    const pooling_layer_create_info_t pool_config = {
        .kernel_width = 2,
        .pooling_operation = POOLING_AVERAGE,
    };
    const linear_layer_create_info_t linear_default_config = {
        .weight_init = linear_weight_init_xavier,
        .bias_init = linear_bias_init_zeros,
    };

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

    model_desc_dump(desc);

    const sequential_model_create_info_t create_info = {
        .desc = desc,
        .max_batch_size = batch_size,
    };
    layer_create(&model, &sequential_model_impl, &create_info, input_shape, batch_size);
    model_desc_destroy(desc);

    return model;
}


dataset_t train_set = NULL, test_set = NULL;
void train_callback(const training_state_t* state)
{
    float test_accuracy = NAN;
    float test_loss = NAN;

    module_test(state->model, test_set, batch_size, state->loss, &test_accuracy, &test_loss);
    LOG_INFO("Epoch %" PRIi32 " | Train loss %f | Train accuracy %5.3f%% | Test loss %f "
        "| Test accuracy %5.3f%%\n",
        state->epoch,
        state->train_loss,
        state->train_accuracy * 100.0f,
        test_loss,
        test_accuracy * 100.0f
    );
}


int main()
{
    /* load the dataset */
    const mnist_create_info_t dataset_config = {
        .path = mnist_path,
        .padding = 0,
    };
    if (dataset_create_train_and_test(&mnist_dataset, &dataset_config, true, &train_set,
                                      &test_set) != 0) {
        LOG_ERROR("There was an error loading the mnist dataset\n");
        return 1;
    }
    LOG_INFO("Successfully loaded mnist\n");


    layer_t lenet5 = create_lenet5(dataset_get_shape(train_set), batch_size);
    LOG_INFO("Created the model. #parameters %d. Start training...\n", module_get_num_params(lenet5));

    /* create the loss */
    Loss loss;
    LossInit(&loss, layer_get_output_shape(lenet5), batch_size, LOSS_FUNCTION_CROSS_ENTROPY);


    module_train(lenet5, train_set, NULL, num_epochs, batch_size, &sgd_optimizer, &optimizer_config,
                 NULL, &loss, train_callback);


    /* Free resources */
    dataset_destroy(train_set);
    dataset_destroy(test_set);
    layer_destroy(lenet5);
    LossDeinit(&loss);

    return 0;
}
