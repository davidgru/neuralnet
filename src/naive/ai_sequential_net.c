
#include "ai_sequential_net.h"

#include <stdio.h>
#include <malloc.h>
#include <string.h>

#include "log.h"

static void set_batch_size(ai_sequential_network_t* net, size_t batch_size)
{
    net->input_layer->mini_batch_size = batch_size;
    for (size_t i = 0; i < net->num_layers; i++)
        net->layers[i]->mini_batch_size = batch_size;
}

static void set_is_training(ai_sequential_network_t* net, uint64_t is_training)
{
    net->input_layer->is_training = is_training;
    for (size_t i = 0; i < net->num_layers; i++)
        net->layers[i]->is_training = is_training;
}


void ai_sequential_network_create(
    ai_sequential_network_t** net,
    ai_input_dims_t* input_dims,
    ai_model_desc_t* desc
)
{
    AI_Layer* input_layer = NULL;
    AI_Layer** layers = (AI_Layer**)calloc(desc->num_layers, sizeof(AI_Layer*));

    /* init dummy input layer */
    AI_InputLayerCreateInfo i_info = {
        .input_width = input_dims->width,
        .input_height = input_dims->height,
        .input_channels = input_dims->channels,
        .batch_size = input_dims->batch_size
    };
    AI_LayerCreateInfo c_info = {
        .type = AI_INPUT_LAYER,
        .create_info = &i_info
    };
    AI_LayerInit(&input_layer, &c_info, 0);

    /* init actual layers */
    for (size_t i = 0; i < desc->num_layers; i++) {
        AI_LayerCreateInfo* current_create_info = &desc->create_infos[i];
        AI_Layer* prev_layer = i == 0 ? input_layer : layers[i - 1];
        AI_LayerInit(&layers[i], current_create_info, prev_layer);
    }

    /* link the layers backward pass. Statically links input and output of adjacent layers
        together */
    for (size_t i = 0; i < desc->num_layers; i++) {
        AI_Layer* prev_layer = i == 0 ? input_layer : layers[i - 1];
        AI_Layer* next_layer = i == desc->num_layers - 1 ? NULL : layers[i + 1];
        AI_LayerLink(layers[i], prev_layer, next_layer);
    }



    size_t input_size = input_dims->batch_size * input_dims->channels * input_dims->height
        * input_dims->width;
    size_t output_size = layers[desc->num_layers - 1]->output_width;
    
    ai_sequential_network_t network = {
        .input_layer = input_layer,
        .layers = layers,
        .num_layers = desc->num_layers,
        .input_size = input_size,
        .output_size = output_size
    };
    *net = (ai_sequential_network_t*)malloc(sizeof(ai_sequential_network_t));
    *(*net) = network;
}


void ai_sequential_network_forward(ai_sequential_network_t* net, float* input)
{
    net->layers[0]->input = input;
    for (size_t i = 0; i < net->num_layers; i++)
        AI_LayerForward(net->layers[i]);
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

    set_batch_size(net, 1);
    set_is_training(net, 0);
    
    for (uint32_t j = 0; j < test_set_size; j++) {
        float* input = test_data + j * net->input_size;
        uint8_t* label = test_labels + j;
        
        ai_sequential_network_forward(net, input);
        
        test_accuracy += AI_LossAccuracy(loss, label);
        test_loss += AI_LossCompute(loss, label);
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

    /* set up the loss and link it with the output layer */
    AI_LossInit(&loss, net->output_size, batch_size, loss_type);
    AI_LossLink(&loss, net->layers[net->num_layers - 1]);


    LOG_TRACE("Performing inital test\n");
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
        set_batch_size(net, batch_size);
        set_is_training(net, 1);
        for (uint32_t j = 0; j < train_dataset_size; j += batch_size) {

            float* input = train_data + j * net->input_size;
            uint8_t* label = train_labels + j;

            /* Forward pass */
            ai_sequential_network_forward(net, input);
            
            /* Loss */
            train_accuracy += AI_LossAccuracy(&loss, label);
            train_loss += AI_LossCompute(&loss, label);

            /* Backward pass */
            AI_LossBackward(&loss, label);
            for (int k = net->num_layers - 1; k >= 0; k--)
                AI_LayerBackward(net->layers[k]);
        }
        train_loss = train_loss / train_dataset_size;
        train_accuracy = train_accuracy / train_dataset_size;
        
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
    AI_LayerDeinit(net->input_layer);
    for (size_t i = 0; i < net->num_layers; i++) {
        AI_LayerDeinit(net->layers[i]);
    }
    free(net);
}
