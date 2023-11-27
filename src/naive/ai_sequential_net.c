
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
    const tensor_shape_t* input_shape,
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


void ai_sequential_network_forward(
    ai_sequential_network_t* net,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t** out_output
)
{
    const tensor_t* current_input = input;
    tensor_t* output = NULL;

    for (size_t i = 0; i < net->num_layers; i++) {
        layer_forward(net->layers[i], forward_kind, current_input, &output);
        current_input = output;
    }

    if (out_output != NULL) {
        *out_output = output;
    }
}


void ai_sequential_network_test(
    ai_sequential_network_t* net,
    dataset_t test_set,
    AI_Loss* loss,
    float* out_accuracy,
    float* out_loss
)
{
    float test_accuracy = 0.0f;
    float test_loss = 0.0f;

    tensor_t* current_inputs = NULL;
    uint8_t* current_targets = NULL;
    dataset_iteration_begin(test_set, 1, false, &current_inputs, &current_targets);

    uint32_t iteration = 0;
    while (current_inputs != NULL) {

        /* forward */
        tensor_t* current_output = NULL;
        ai_sequential_network_forward(net, LAYER_FORWARD_INFERENCE, current_inputs, &current_output);

        /* metrics */
        test_accuracy += AI_LossAccuracy(loss, current_output, current_targets);
        test_loss += AI_LossCompute(loss, current_output, current_targets);
    
        /* prepare next round */
        dataset_iteration_next(test_set, &current_inputs, &current_targets);

        iteration++;
    }

    const size_t dataset_size = dataset_get_shape(test_set)->dims[TENSOR_BATCH_DIM];
    *out_accuracy = test_accuracy / (float)dataset_size;
    *out_loss = test_loss / (float)dataset_size;
}


void ai_sequential_network_train(
    ai_sequential_network_t* net,
    dataset_t train_set,
    dataset_t test_set,
    size_t num_epochs,
    size_t batch_size,
    const optimizer_impl_t* optimizer_impl,
    const optimizer_config_t* optimizer_config,
    AI_LossFunctionEnum loss_type,
    ai_training_callback_t callback
)
{
    AI_Loss loss;

    /* set up the optimizer */
    optimizer_t optimizer;
    optimizer_create(&optimizer, optimizer_impl, optimizer_config);

    for (size_t i = 0; i < net->num_layers; i++) {
        layer_param_ref_list_t layer_param_refs;
        layer_get_param_refs(net->layers[i], &layer_param_refs);
        optimizer_add_params(optimizer, &layer_param_refs);
    }

    tensor_shape_t train_input_shape = net->input_shape;
    train_input_shape.dims[TENSOR_BATCH_DIM] = batch_size;

    size_t input_size = tensor_size_from_shape(&train_input_shape);
    size_t output_size = tensor_size_from_shape(&net->output_shape);

    AI_LossInit(&loss, &net->output_shape, batch_size, loss_type);


    LOG_TRACE("Performing initial test\n");
    float test_accuracy;
    float test_loss;
    ai_sequential_network_test(net, test_set, &loss,
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

        tensor_t* current_inputs = NULL;
        uint8_t* current_targets = NULL;
        dataset_iteration_begin(train_set, batch_size, true, &current_inputs, &current_targets);

        while(current_inputs != NULL) {
            
            tensor_t* output;
            ai_sequential_network_forward(net, LAYER_FORWARD_TRAINING, current_inputs, &output);

            /* Loss */
            train_accuracy += AI_LossAccuracy(&loss, output, current_targets);
            train_loss += AI_LossCompute(&loss, output, current_targets);

            /* Backward pass */
            tensor_t* gradient;
            AI_LossBackward(&loss, output, current_targets, &gradient);

            for (int32_t k = net->num_layers - 1; k >= 0; k--) {
                tensor_t* next_gradient;
                layer_backward(net->layers[k], gradient, &next_gradient);
                gradient = next_gradient;
            }

            optimizer_step(optimizer);

            dataset_iteration_next(train_set, &current_inputs, &current_targets);
        }

        const size_t train_set_size = dataset_get_shape(train_set)->dims[TENSOR_BATCH_DIM];
        train_loss = train_loss / (float)train_set_size;
        train_accuracy = train_accuracy / (float)train_set_size;


        /* Test */
        ai_sequential_network_test(net, test_set, &loss,
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
