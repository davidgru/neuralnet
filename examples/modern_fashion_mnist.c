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

#include "optimizer/adam.h"
#include "optimizer/rmsprop.h"
#include "optimizer/sgd.h"

#include "sequential/model_desc.h"
#include "sequential/sequential_model.h"

#include "dataset/dataset.h"
#include "dataset/mnist.h"

#include "augment/augment_pipeline.h"
#include "augment/image_flip.h"

#include "util/training_utils.h"

#include "config_info.h"
#include "log.h"
#include "tensor.h"


layer_t create_lenet5(const tensor_shape_t* input_shape, float dropout_rate, bool use_batchnorm, size_t batch_size)
{
    model_desc_t* desc = NULL;
    layer_t model = NULL;


    /* Allocate resources for the model descriptor. */
    model_desc_create(&desc);

    model_desc_add_convolutional_layer(desc, 6, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_convolutional_layer(desc, 16, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_pooling_layer(desc, 2, 1, 0, POOLING_MAX);

    model_desc_add_convolutional_layer(desc, 32, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_convolutional_layer(desc, 32, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_pooling_layer(desc, 2, 1, 0, POOLING_MAX);

    model_desc_add_linear_layer(desc, 512, linear_weight_init_he, linear_bias_init_zeros);
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    if (dropout_rate > 0.0f) {
        model_desc_add_dropout_layer(desc, dropout_rate);
    }

    model_desc_add_linear_layer(desc, 194, linear_weight_init_he, linear_bias_init_zeros);
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    if (dropout_rate > 0.0f) {
        model_desc_add_dropout_layer(desc, dropout_rate);
    }

    model_desc_add_linear_layer(desc, 10, linear_weight_init_he, linear_bias_init_zeros);


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


augment_pipeline_t setup_augment_pipeline()
{
    image_flip_config_t flip_config = {
        .horizontal_flip_prob = 0.5f,
        .vertical_flip_prob = 0.0f,
    };


    augment_pipeline_config_entry_t pipeline_entries[] = {
        { .impl = &aug_image_flip, .config = &flip_config },
    };

    augment_pipeline_config_t pipeline_config = {
        .entries = pipeline_entries,
        .num_entries = sizeof(pipeline_entries) / sizeof(*pipeline_entries),
    };

    augment_pipeline_t augment_pipeline = NULL;
    augment_pipeline_create(&augment_pipeline, &pipeline_config);
    return augment_pipeline;
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
    const char* mnist_path = "/home/david/projects/neuralnet/datasets/fashion_mnist";


    /* When training on mnist with this configuration, the model should reach an accuracy of 90%+
        after one epoch and an accuracy of ~98.5% after 10 epochs */
    size_t num_epochs = 10000;
    size_t batch_size = 32;
    LossFunctionEnum loss_type = LOSS_FUNCTION_CROSS_ENTROPY;
    /* use sgd optimizer */
    const optimizer_impl_t* optimizer = &adam_optimizer;
    // sgd_config_t optimizer_config = {
    //     .learning_rate = 1e-2f,
    //     .weight_reg_kind = SGD_WEIGHT_REG_L2,
    //     .weight_reg_strength = 1e-4,
    // };
    // rmsprop_config_t optimizer_config = {
    //     .learning_rate = 1e-3f,
    //     .gamma = 0.9f,
    //     .weight_reg_kind = WEIGHT_REG_L2,
    //     .weight_reg_strength = 1e-4f,
    // };
    adam_config_t optimizer_config = {
        .learning_rate = 1e-4f,
        .gamma1 = 0.999f,
        .gamma2 = 0.9f,
        .weight_reg_kind = WEIGHT_REG_L2,
        .weight_reg_strength = 1e-4f,
    };
    /* reduce learning rate after 5 epochs without progress on training loss */
    size_t reduce_learning_rate_after = 5;
    bool use_batchnorm = true;
    float dropout_rate = 0.5f;
    bool augment = true;


    /* Verify the compile time configuration. For example, that avx is used */
    dump_compile_time_config();


    /* Load mnist with a padding of two because lenet expects 32x32 input and the naive
        convolutional layers do not support padding at this time. */
    dataset_t train_set = load_mnist(mnist_path, MNIST_TRAIN_SET);
    dataset_t test_set = load_mnist(mnist_path, MNIST_TEST_SET);

    if (train_set == NULL || test_set == NULL) {
        LOG_ERROR("There was an error loading the mnist dataset\n");
        return 1;
    }
    LOG_INFO("Successfully loaded mnist\n");


    augment_pipeline_t augment_pipeline = augment ? setup_augment_pipeline() : NULL;
    if (augment && augment_pipeline == NULL) {
        LOG_ERROR("There was an error setting up the augmentation pipeline\n");
        return 1;
    }
    LOG_INFO("Successfully set up the augmentation pipeline\n");

    layer_t lenet5 = create_lenet5(dataset_get_shape(train_set), dropout_rate, use_batchnorm, batch_size);
    LOG_INFO("Created the model. Start training with learning rate %.2e ...\n", optimizer_config.learning_rate);


    module_train(lenet5, train_set, test_set, augment_pipeline, num_epochs, batch_size, optimizer,
        &optimizer_config, loss_type, reduce_learning_rate_after, train_callback);


    /* Free resources */
    layer_destroy(lenet5);
    dataset_destroy(train_set);
    dataset_destroy(test_set);
    if (augment_pipeline != NULL) {
        augment_pipeline_destroy(augment_pipeline);
    }

    return 0;
}
