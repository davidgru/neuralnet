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
#include "augment/random_crop.h"
#include "augment/color_augment.h"

#include "util/training_utils.h"

#include "config_info.h"
#include "log.h"
#include "tensor.h"
#include "context.h"


layer_t create_mlp(const tensor_shape_t* input_shape, float dropout_rate, bool use_batchnorm, size_t batch_size)
{
    model_desc_t* desc;
    layer_t model;

    model_desc_create(&desc);

    model_desc_add_linear_layer(desc, 100, linear_weight_init_xavier, linear_bias_init_zeros);
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_SIGMOID);

    model_desc_add_linear_layer(desc, 10, linear_weight_init_xavier, linear_bias_init_zeros);

    /* Print a model overview to stdout. */
    model_desc_dump(desc);

    const sequential_model_create_info_t config = {
        .desc = desc,
        .max_batch_size = batch_size
    };

    layer_create(&model, &sequential_model_impl, &config, input_shape, batch_size);

    /* Model desc not needed anymore */
    model_desc_destroy(desc);

    return model;
}

/* 92.13% */
layer_t create_lenet5(const tensor_shape_t* input_shape, float dropout_rate, bool use_batchnorm, size_t batch_size)
{
    model_desc_t* desc = NULL;
    layer_t model = NULL;

    const float bn_momentum = 0.9f;
    const float bn_eps = 1e-8f;

    /* Allocate resources for the model descriptor. */
    model_desc_create(&desc);

    model_desc_add_convolutional_layer(desc, 6, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_convolutional_layer(desc, 6, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_pooling_layer(desc, 2, 1, 0, POOLING_MAX);

    model_desc_add_convolutional_layer(desc, 16, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_convolutional_layer(desc, 16, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_pooling_layer(desc, 2, 1, 0, POOLING_MAX);

    model_desc_add_linear_layer(desc, 120, linear_weight_init_he, linear_bias_init_zeros);
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    if (dropout_rate > 0.0f) {
        model_desc_add_dropout_layer(desc, dropout_rate);
    }

    model_desc_add_linear_layer(desc, 84, linear_weight_init_he, linear_bias_init_zeros);
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

/* 92.61% */
layer_t create_lenet5_wider(const tensor_shape_t* input_shape, float dropout_rate, bool use_batchnorm, size_t batch_size)
{
    model_desc_t* desc = NULL;
    layer_t model = NULL;

    const float bn_momentum = 0.9f;
    const float bn_eps = 1e-8f;

    /* Allocate resources for the model descriptor. */
    model_desc_create(&desc);

    model_desc_add_convolutional_layer(desc, 32, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_convolutional_layer(desc, 32, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_pooling_layer(desc, 2, 1, 0, POOLING_MAX);

    model_desc_add_convolutional_layer(desc, 32, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_convolutional_layer(desc, 32, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_pooling_layer(desc, 2, 1, 0, POOLING_MAX);

    model_desc_add_linear_layer(desc, 256, linear_weight_init_he, linear_bias_init_zeros);
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    if (dropout_rate > 0.0f) {
        model_desc_add_dropout_layer(desc, dropout_rate);
    }

    model_desc_add_linear_layer(desc, 128, linear_weight_init_he, linear_bias_init_zeros);
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


/* 92.91% */
layer_t create_lenet5_wider_wider(
    const tensor_shape_t* input_shape, float dropout_rate, bool use_batchnorm, size_t batch_size)
{
    model_desc_t* desc = NULL;
    layer_t model = NULL;

    const float bn_momentum = 0.9f;
    const float bn_eps = 1e-8f;

    /* Allocate resources for the model descriptor. */
    model_desc_create(&desc);

    model_desc_add_convolutional_layer(desc, 64, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_convolutional_layer(desc, 64, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_pooling_layer(desc, 2, 1, 0, POOLING_MAX);

    model_desc_add_convolutional_layer(desc, 64, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_convolutional_layer(desc, 64, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_pooling_layer(desc, 2, 1, 0, POOLING_MAX);


    model_desc_add_linear_layer(desc, 1024, linear_weight_init_he, linear_bias_init_zeros);
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    if (dropout_rate > 0.0f) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
    }

    model_desc_add_linear_layer(desc, 256, linear_weight_init_he, linear_bias_init_zeros);
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    if (dropout_rate > 0.0f) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
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


/* crop1: 93.51% */
layer_t create_lenet5_wider_less_linear(const tensor_shape_t* input_shape, float dropout_rate, bool use_batchnorm, size_t batch_size)
{
    model_desc_t* desc = NULL;
    layer_t model = NULL;

    const float bn_momentum = 0.9f;
    const float bn_eps = 1e-8f;

    /* Allocate resources for the model descriptor. */
    model_desc_create(&desc);

    model_desc_add_convolutional_layer(desc, 64, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_convolutional_layer(desc, 64, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_pooling_layer(desc, 2, 1, 0, POOLING_MAX);

    model_desc_add_convolutional_layer(desc, 64, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_convolutional_layer(desc, 64, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_pooling_layer(desc, 2, 1, 0, POOLING_MAX);


    model_desc_add_linear_layer(desc, 512, linear_weight_init_he, linear_bias_init_zeros);
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    if (dropout_rate > 0.0f) {
        model_desc_add_dropout_layer(desc, dropout_rate);
    }

    model_desc_add_linear_layer(desc, 128, linear_weight_init_he, linear_bias_init_zeros);
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


layer_t create_lenet5_wider_less_linear_wider(const tensor_shape_t* input_shape, float dropout_rate, bool use_batchnorm, size_t batch_size)
{
    model_desc_t* desc = NULL;
    layer_t model = NULL;

    const float bn_momentum = 0.9f;
    const float bn_eps = 1e-8f;

    /* Allocate resources for the model descriptor. */
    model_desc_create(&desc);

    model_desc_add_convolutional_layer(desc, 128, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_convolutional_layer(desc, 128, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_pooling_layer(desc, 2, 1, 0, POOLING_MAX);

    model_desc_add_convolutional_layer(desc, 128, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_convolutional_layer(desc, 128, 3, 1, 1, conv_weight_init_he, conv_bias_init_zeros);
    if (use_batchnorm) {
        model_desc_add_batch_norm_layer(desc, bn_momentum, bn_eps);
    }
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    model_desc_add_pooling_layer(desc, 2, 1, 0, POOLING_MAX);


    model_desc_add_linear_layer(desc, 1024, linear_weight_init_he, linear_bias_init_zeros);
    model_desc_add_activation_layer(desc, ACTIVATION_FUNCTION_RELU);
    if (dropout_rate > 0.0f) {
        model_desc_add_dropout_layer(desc, dropout_rate);
    }

    model_desc_add_linear_layer(desc, 256, linear_weight_init_he, linear_bias_init_zeros);
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
    const image_flip_config_t flip_config = {
        .horizontal_flip_prob = 0.5f,
        .vertical_flip_prob = 0.0f,
    };

    const random_crop_config_t crop_config = {
        .padding = 2,
    };

    const color_augment_config_t color_config = {
        .brightness_augment_prob = 0.5f,
        .brightness_std = 0.1f,
        .contrast_augment_prob = 0.5f,
        .contrast_std = 0.1f,
        .contrast_midpoint = 0.0f /* since images are normalized */
    };


    const augment_pipeline_config_entry_t pipeline_entries[] = {
        { .impl = &aug_image_flip, .config = &flip_config },
        { .impl = &aug_random_crop, .config = &crop_config },
        // { .impl = &aug_color, .config = &color_config }
    };

    const augment_pipeline_config_t pipeline_config = {
        .entries = pipeline_entries,
        .num_entries = sizeof(pipeline_entries) / sizeof(*pipeline_entries),
    };

    augment_pipeline_t augment_pipeline = NULL;
    augment_pipeline_create(&augment_pipeline, &pipeline_config);
    return augment_pipeline;
}


dataset_t train_set = NULL, test_set = NULL;
uint32_t load_mnist(const char* path, dataset_t* train, dataset_t* test)
{
    /* load train set and calculate dataset mean and variance for normalization */
    const mnist_create_info_t train_config = {
        .path = path,
        .dataset_kind = TRAIN_SET,
        .padding = 0
    };
    uint32_t status = dataset_create(train, &mnist_dataset, &train_config, true, NULL);
    if (status != 0) {
        return status;
    }
    const dataset_statistics_t* train_statistics = dataset_get_statistics(*train);
    LOG_INFO("Dataset mean %f stddev %f\n", train_statistics->mean, train_statistics->stddev);

    /* load test set and use mean and variance of train set for normalization */
    const mnist_create_info_t test_config = {
        .path = path,
        .dataset_kind = TEST_SET,
        .padding = 0
    };
    return dataset_create(test, &mnist_dataset, &test_config, true, train_statistics);
}


Loss loss;
size_t batch_size = 32;
void train_callback(const training_state_t* state)
{
    float test_accuracy;
    float test_loss;

    if (state->epoch > 0) {
        
        if (state->epoch % 5 == 0) {
            module_test_10crop(state->model, test_set, batch_size, 2, NULL, &test_accuracy, &test_loss);
        } else {
            module_test(state->model, test_set, batch_size, &loss, &test_accuracy, &test_loss);
        }
    }
    printf("Epoch %" PRIi32 " | Train loss %f | Train accuracy %5.2f%% | Test loss %f "
        "| Test accuracy %s %5.2f%% | lr %.2e\n",
        state->epoch,
        state->train_loss,
        state->train_accuracy * 100.0f,
        test_loss,
        state->epoch % 5 == 0 ? "(10crop)" : "",
        test_accuracy * 100.0f,
        optimizer_get_learning_rate(state->optimizer)
    );
}

#include <math.h>


float linear_lr_schedule(const training_state_t* state)
{
    static const float initial_lr = 0.5f;
    static const float final_lr = 0.01f;

    static const size_t decay_begin = 0;
    static const size_t decay_end = 20;

    static const int32_t reduce_lr_after = 7;
    static const float reduce_lr_by_factor = 10.0f;

    static float min_train_loss = INFINITY;
    static int32_t min_train_loss_epoch = 0;

    if (state->train_loss < min_train_loss) {
        min_train_loss = state->train_loss;
        min_train_loss_epoch = state->epoch;
    }

    if (state->epoch < decay_begin) {
        return initial_lr;
    } else if (state->epoch <= decay_end) {
        return final_lr + (initial_lr - final_lr) * (decay_end - state->epoch) / (decay_end - decay_begin);
    } else {
        if (reduce_lr_after != 0 && (int32_t)state->epoch - min_train_loss_epoch >= reduce_lr_after) {
            min_train_loss = state->train_loss;
            min_train_loss_epoch = state->epoch;
            return optimizer_get_learning_rate(state->optimizer) / reduce_lr_by_factor;
        } else {
            return optimizer_get_learning_rate(state->optimizer);
        }
    }
}


int main()
{
    /* set to location of mnist or fashion_mnist root folder */
    const char* mnist_path = "/home/david/projects/neuralnet/datasets/fashion_mnist";


    /* When training on mnist with this configuration, the model should reach an accuracy of 90%+
        after one epoch and an accuracy of ~98.5% after 10 epochs */
    size_t num_epochs = 10000;
    LossFunctionEnum loss_type = LOSS_FUNCTION_CROSS_ENTROPY;
    /* use sgd optimizer */
    const optimizer_impl_t* optimizer = &sgd_optimizer;
    sgd_config_t optimizer_config = {
        .learning_rate = 0.1f,
        .weight_reg_kind = SGD_WEIGHT_REG_L2,
        .weight_reg_strength = 1e-4,
    };
    // rmsprop_config_t optimizer_config = {
    //     .learning_rate = 1e-3f,
    //     .gamma = 0.9f,
    //     .weight_reg_kind = WEIGHT_REG_L2,
    //     .weight_reg_strength = 1e-4f,
    // };
    // adam_config_t optimizer_config = {
    //     .learning_rate = 1e-3f,
    //     .gamma1 = 0.999f,
    //     .gamma2 = 0.9f,
    //     .weight_reg_kind = WEIGHT_REG_L2,
    //     .weight_reg_strength = 1e-4f,
    // };
    bool use_batchnorm = true;
    float dropout_rate = 0.0f;
    bool augment = true;


    /* Verify the compile time configuration. For example, that avx is used */
    dump_compile_time_config();


    /* Initialize the backend context. Only needed for the oneDNN backend */
    if (backend_context_init() != 0) {
        LOG_ERROR("Failed to initialize the backend context\n");
        return 1;
    }

    
    if (load_mnist(mnist_path, &train_set, &test_set) != 0) {
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

    layer_t model = create_lenet5_wider_less_linear_wider(dataset_get_shape(train_set), dropout_rate, use_batchnorm, batch_size);
    size_t num_params = module_get_num_params(model);
    LOG_INFO("Created the model. #params = %zu\n", num_params);

    /* initialize loss */
    LossInit(&loss, layer_get_output_shape(model), batch_size, loss_type);

    LOG_INFO("Start training with learning rate %.2e\n", optimizer_config.learning_rate);
    module_train(model, train_set, augment_pipeline, num_epochs, batch_size, optimizer,
        &optimizer_config, linear_lr_schedule, &loss, train_callback);


    /* Free resources */
    layer_destroy(model);
    dataset_destroy(train_set);
    dataset_destroy(test_set);
    if (augment_pipeline != NULL) {
        augment_pipeline_destroy(augment_pipeline);
    }
    LossDeinit(&loss);

    return 0;
}
