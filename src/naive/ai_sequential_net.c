
#include "ai_sequential_net.h"

#include <stdio.h>
#include <stddef.h>
#include <malloc.h>
#include <string.h>

#include "log.h"

// static void set_is_training(ai_sequential_network_t* net, uint64_t is_training)
// {
//     net->input_layer->is_training = is_training;
//     for (size_t i = 0; i < net->num_layers; i++)
//         net->layers[i]->is_training = is_training;
// }


void ai_sequential_network_create(
    ai_sequential_network_t** net,
    tensor_shape_t* input_shape,
    size_t max_batch_size,
    ai_model_desc_t* desc
)
{
    layer_t* layers = (layer_t*)calloc(desc->num_layers, sizeof(layer_t));


    /* initialize the layers */
    const tensor_shape_t* output_shape = input_shape;

    for (size_t i = 0; i < desc->num_layers; i++) {
        AI_LayerCreateInfo* create_info = &desc->create_infos[i];

        layer_create(&layers[i], create_info, output_shape, max_batch_size);
        output_shape = layer_get_output_shape(layers[i]);
    }


    ai_sequential_network_t network = {
        .layers = layers,
        .num_layers = desc->num_layers,
        .input_shape = *input_shape,
        .output_shape = *output_shape,
    };

    *net = (ai_sequential_network_t*)malloc(sizeof(ai_sequential_network_t));
    *(*net) = network;
}


void ai_sequential_network_forward(ai_sequential_network_t* net, const tensor_t* input, tensor_t** out_output)
{
    const tensor_t* current_input = input;
    tensor_t* output = NULL;

    for (size_t i = 0; i < net->num_layers; i++) {
        layer_forward(net->layers[i], current_input, &output);
        current_input = output;
    }

    if (out_output != NULL) {
        *out_output = output;
    }
}


void ai_sequential_network_test(
    ai_sequential_network_t* net,
    float* test_data,
    uint8_t* test_labels,
    size_t test_set_size,
    AI_Loss* loss,
    float* out_accuracy,
    float* out_loss
)
{
    float test_accuracy = 0.0f;
    float test_loss = 0.0f;


    tensor_shape_t test_input_shape = net->input_shape;
    test_input_shape.dims[TENSOR_BATCH_DIM] = 1; /* use batch size of 1 for testing for now */

    size_t input_size = tensor_size_from_shape(&test_input_shape);


    for (uint32_t j = 0; j < test_set_size; j++) {
        float* input = test_data + j * input_size;
        uint8_t* label = test_labels + j;

        tensor_t input_tensor;
        tensor_from_memory(&input_tensor, &test_input_shape, input);

        tensor_t* output;
        ai_sequential_network_forward(net, &input_tensor, &output);

        test_accuracy += AI_LossAccuracy(loss, output, label);
        test_loss += AI_LossCompute(loss, output, label);
    }

    *out_accuracy = test_accuracy / (float)test_set_size;
    *out_loss = test_loss / (float)test_set_size;
}


void ai_sequential_network_train(
    ai_sequential_network_t* net,
    float* train_data,
    float* test_data,
    uint8_t* train_labels,
    uint8_t* test_labels,
    size_t train_dataset_size,
    size_t test_dataset_size,
    size_t num_epochs,
    float learning_rate,
    size_t batch_size,
    AI_LossFunctionEnum loss_type,
    ai_training_callback_t callback
)
{
    AI_Loss loss;

    tensor_shape_t train_input_shape = net->input_shape;
    train_input_shape.dims[TENSOR_BATCH_DIM] = batch_size;

    size_t input_size = tensor_size_from_shape(&train_input_shape);
    size_t output_size = tensor_size_from_shape(&net->output_shape);

    AI_LossInit(&loss, &net->output_shape, batch_size, loss_type);


    LOG_TRACE("Performing initial test\n");
    float test_accuracy;
    float test_loss;
    ai_sequential_network_test(net, test_data, test_labels, test_dataset_size, &loss,
        &test_accuracy, &test_loss);

    if (callback) {
        ai_training_info_t progress_info = {
            .epoch = 0,
            .train_loss = 0.0f,
            .train_accuracy = 0.0f,
            .test_loss = test_loss,
            .test_accuracy = test_accuracy
        };
        callback(&progress_info);
    }

    // for (size_t i = 0; i < net->num_layers; i++)
    //     AI_LayerInfo(net->layers[i]);


    LOG_TRACE("Starting training loop\n");
    for (uint32_t i = 0; i < num_epochs; i++) {
        LOG_TRACE("Epoch: %d\n", i + 1);

        float train_loss = 0.0f;
        float train_accuracy = 0.0f;

        /* Train one epoch */
        // set_batch_size(net, batch_size);
        // set_is_training(net, 1);
        for (uint32_t j = 0; j < train_dataset_size; j += batch_size) {

            float* input = train_data + j * input_size;
            uint8_t* label = train_labels + j;

            /* Forward pass */
            tensor_t input_tensor;
            tensor_from_memory(&input_tensor, &train_input_shape, input);

            tensor_t* output;
            ai_sequential_network_forward(net, &input_tensor, &output);

            /* Loss */
            train_accuracy += AI_LossAccuracy(&loss, output, label);
            train_loss += AI_LossCompute(&loss, output, label);

            /* Backward pass */
            tensor_t* gradient;
            AI_LossBackward(&loss, output, label, &gradient);

            for (int32_t k = net->num_layers - 1; k >= 0; k--) {
                tensor_t* next_gradient;
                layer_backward(net->layers[k], gradient, &next_gradient);
                gradient = next_gradient;
            }
        }
        train_loss = train_loss * batch_size / train_dataset_size;
        train_accuracy = train_accuracy * batch_size / train_dataset_size;
        
        /* Test */
        ai_sequential_network_test(net, test_data, test_labels, test_dataset_size, &loss,
            &test_accuracy, &test_loss);

        if (callback) {
            ai_training_info_t progress_info = {
                .epoch = i + 1,
                .train_loss = train_loss,
                .train_accuracy = train_accuracy,
                .test_loss = test_loss,
                .test_accuracy = test_accuracy
            };
            callback(&progress_info);
        }

        // for (size_t i = 0; i < net->num_layers; i++)
        //     AI_LayerInfo(net->layers[i]);

    }
}


void ai_sequential_network_destroy(ai_sequential_network_t* net)
{
    for (size_t i = 0; i < net->num_layers; i++) {
        layer_destroy(net->layers[i]);
    }
    free(net);
}
