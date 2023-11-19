
#include "ai_net.h"

#include <stdio.h>
#include <malloc.h>
#include <string.h>

#include "log.h"

static void set_batch_size(AI_Net* net, size_t batch_size)
{
    net->input_layer->mini_batch_size = batch_size;
    for (size_t i = 0; i < net->num_layers; i++)
        net->layers[i]->mini_batch_size = batch_size;
}

static void set_is_training(AI_Net* net, uint64_t is_training)
{
    net->input_layer->is_training = is_training;
    for (size_t i = 0; i < net->num_layers; i++)
        net->layers[i]->is_training = is_training;
}

void AI_NetInit(AI_Net* net, size_t input_width, size_t input_height, size_t input_channels, size_t batch_size, AI_LayerCreateInfo* create_infos, size_t num_layers)
{
    net->num_layers = num_layers;
    net->layers = (AI_Layer**)malloc(num_layers * sizeof(AI_Layer*));
    
    // Create the input layer
    AI_InputLayerCreateInfo i_info = { input_width, input_height, input_channels, batch_size };
    AI_LayerCreateInfo c_info = { AI_INPUT_LAYER, &i_info };
    AI_LayerInit(&net->input_layer, &c_info, 0);

    // Create the layers
    AI_LayerInit(&net->layers[0], &create_infos[0], net->input_layer);
    for (size_t i = 1; i < num_layers; i++)
        AI_LayerInit(&net->layers[i], &create_infos[i], net->layers[i - 1]);
    AI_LossInit(&net->loss, 10, 1, AI_LOSS_FUNCTION_MSE);

    // Link the layers
    AI_LayerLink(net->layers[0], 0, net->layers[1]);
    for (size_t i = 1; i < num_layers - 1; i++)
        AI_LayerLink(net->layers[i], net->layers[i - 1], net->layers[i + 1]);
    AI_LayerLink(net->layers[num_layers - 1], net->layers[num_layers - 2], 0);
    AI_LossLink(&net->loss, net->layers[num_layers - 1]);

    net->input_size = input_width * input_height * input_channels;
    net->output_size = net->loss.size;
}


void AI_NetForward(AI_Net* net, float* input)
{
    net->layers[0]->input = input;
    for (size_t i = 0; i < net->num_layers; i++)
        AI_LayerForward(net->layers[i]);
}


void AI_NetTrain(AI_Net* net, float* train_data, float* test_data, uint8_t* train_labels, uint8_t* test_labels, size_t train_dataset_size, size_t test_dataset_size, size_t num_epochs, float learning_rate, size_t batch_size, AI_TrainCallback callback)
{
    AI_TrainingProgress progress_info;

    memset(&progress_info, 0, sizeof(progress_info));

    float test_loss = 0.0f;
    float test_accuracy = 0.0f;

    LOG_TRACE("Performing inital test\n");

    // Initial Test
    set_batch_size(net, 1);
    set_is_training(net, 0);
    for (uint32_t j = 0; j < test_dataset_size; j++) {
        
        LOG_TRACE("Retrieving input/target pair\n");
        
        float* input = test_data + j * net->input_size;
        uint8_t* label = test_labels + j;

        
        LOG_TRACE("Initiating forward pass\n");
        
        AI_NetForward(net, input);
        

        test_accuracy += AI_LossAccuracy(&net->loss, label);
        test_loss += AI_LossCompute(&net->loss, label);
    }
    test_accuracy = test_accuracy * 100.0f / test_dataset_size;
    test_loss /= test_dataset_size;


    LOG_TRACE("Registering callbacks\n");


    if (callback) {
        progress_info.epoch = -1;
        progress_info.train_loss = 0.0f;
        progress_info.train_accuracy = 0.0f;
        progress_info.test_loss = test_loss;
        progress_info.test_accuracy = test_accuracy;

        callback(&progress_info);
    }


    LOG_TRACE("Starting training loop\n");


    for (uint32_t i = 0; i < num_epochs; i++) {

        LOG_TRACE("Epoch: %d\n", i + 1);


        float train_loss = 0.0f;
        float train_accuracy = 0.0f;

        test_loss = 0.0f;
        test_accuracy = 0.0f;

        // Train one epoch
        set_batch_size(net, batch_size);
        set_is_training(net, 1);
        for (uint32_t j = 0; j < train_dataset_size; j += batch_size) {

            LOG_TRACE("Retrieving input/target pair\n");
            float* input = train_data + j * net->input_size;
            uint8_t* label = train_labels + j;

            // Forward pass

            LOG_TRACE("Performing forward pass\n");
            AI_NetForward(net, input);
            
            // Evaluation
            train_accuracy += AI_LossAccuracy(&net->loss, label);
            train_loss += AI_LossCompute(&net->loss, label);

            // Backward pass
            AI_LossBackward(&net->loss, label);
            for (size_t k = net->num_layers - 1; k >= 1; k--)
                AI_LayerBackward(net->layers[k]);
        }
        train_loss = train_loss / train_dataset_size;
        train_accuracy = train_accuracy * 100.0f / train_dataset_size;
        
        // Test
        set_batch_size(net, 1);
        set_is_training(net, 0);
        for (uint32_t j = 0; j < test_dataset_size; j++) {
            float* input = test_data + j * net->input_size;
            uint8_t* label = test_labels + j;
            AI_NetForward(net, input);
            test_accuracy += AI_LossAccuracy(&net->loss, label);
            test_loss += AI_LossCompute(&net->loss, label);
        }
        test_accuracy = test_accuracy * 100.0f / test_dataset_size;
        test_loss /= test_dataset_size;

        if (callback) {
            progress_info.epoch = i;
            progress_info.train_loss = train_loss;
            progress_info.train_accuracy = train_accuracy;
            progress_info.test_loss = test_loss;
            progress_info.test_accuracy = test_accuracy;

            callback(&progress_info);
        }
    }
}

void AI_NetDeinit(AI_Net* net)
{
    AI_LayerDeinit(net->input_layer);
    for (size_t i = 0; i < net->num_layers; i++)
        AI_LayerDeinit(net->layers[i]);
    AI_LossDeinit(&net->loss);
}
