/**
 * @file two_layer_mlp_mnist.c
 * @brief Train a two-layered MLP on the MNIST dataset
 * 
 * This example illustrates how this library can be used to implement a two layer deep MLP and
 * trains it on the MNIST dataset.
 */


#include <stdio.h>
#include <inttypes.h>

#include "core/layer.h"
#include "core/loss.h"
#include "core/optimizer.h"

#include "dataset/dataset.h"
#include "dataset/mnist.h"

#include "sequential/model_desc.h"
#include "sequential/sequential_model.h"

#include "optimizer/sgd.h"

#include "util/training_utils.h"

#include "log.h"
#include "config_info.h"

#include "tensor.h"
#include "context.h"
#include "tensor_impl.h"


layer_t create_mlp(const tensor_shape_t* input_shape, size_t batch_size)
{
    model_desc_t* desc;
    layer_t model;

    model_desc_create(&desc);

    model_desc_add_linear_layer(desc, 300, linear_weight_init_xavier, linear_bias_init_zeros);
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


int main()
{
    /* set to location of mnist or fashion_mnist root folder */
    const char* mnist_path = "/home/david/projects/neuralnet/datasets/mnist";


    const size_t num_epochs = 10;
    const size_t batch_size = 32;
    const LossFunctionEnum loss_type = LOSS_FUNCTION_CROSS_ENTROPY;
    const optimizer_impl_t* optimizer = &sgd_optimizer;
    const sgd_config_t optimizer_config = {
        .learning_rate = 0.5f,
        .weight_reg_kind = WEIGHT_REG_NONE
    };


    /* Verify the compile time configuration. For example, that avx is used */
    dump_compile_time_config();


    /* Initialize the backend context. Only needed for the oneDNN backend */
    if (backend_context_init() != 0) {
        LOG_ERROR("Failed to initialize the backend context\n");
        return 1;
    }


    dataset_t train_set = load_mnist(mnist_path, MNIST_TRAIN_SET);
    dataset_t test_set = load_mnist(mnist_path, MNIST_TEST_SET);
    if (train_set == NULL || test_set == NULL) {
        LOG_ERROR("There was an error loading the mnist dataset\n");
        return 1;
    }
    LOG_INFO("Successfully loaded mnist\n");


    layer_t mlp = create_mlp(dataset_get_shape(train_set), batch_size);
    if (mlp == NULL) {
        LOG_ERROR("There was an error creating the model\n");
        return 1;
    }
    LOG_INFO("Created the model. Start training...\n");


    /* Training loop */
    module_train(mlp, train_set, test_set, NULL, num_epochs, batch_size, optimizer,
    &optimizer_config, loss_type, 0, train_callback);


    /* Free resources */
    layer_destroy(mlp);
    dataset_destroy(train_set);
    dataset_destroy(test_set);
    
    return 0;
}
