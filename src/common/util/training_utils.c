#include <stdbool.h>
#include <malloc.h>
#include <math.h>

#include "log.h"

#include "training_utils.h"


void module_test(
    layer_t layer,
    dataset_t test_set,
    size_t batch_size,
    Loss* loss,
    float* out_accuracy,
    float* out_loss
)
{
    float test_accuracy = 0.0f;
    float test_loss = 0.0f;

    tensor_t* current_inputs = NULL;
    uint8_t* current_targets = NULL;
    dataset_iteration_begin(test_set, batch_size, false, &current_inputs, &current_targets);

    uint32_t iteration = 0;
    while (current_inputs != NULL) {

        /* forward */
        tensor_t* current_output = NULL;
        layer_forward(layer, LAYER_FORWARD_INFERENCE, current_inputs, &current_output);

        /* metrics */
        test_accuracy += LossAccuracy(loss, current_output, current_targets);
        test_loss += LossCompute(loss, current_output, current_targets);
    
        /* prepare next round */
        dataset_iteration_next(test_set, &current_inputs, &current_targets);

        iteration++;
    }

    const size_t dataset_size = tensor_shape_get_dim(dataset_get_shape(test_set), TENSOR_BATCH_DIM);
    *out_accuracy = test_accuracy / (float)dataset_size;
    *out_loss = test_loss / (float)dataset_size;
}


void module_train(
    layer_t layer,
    dataset_t train_set,
    dataset_t test_set,
    augment_pipeline_t augment_pipeline,
    size_t num_epochs,
    size_t batch_size,
    const optimizer_impl_t* optimizer_impl,
    const optimizer_config_t* optimizer_config,
    LossFunctionEnum loss_type,
    size_t reduce_lr_after,
    training_callback_t callback
)
{
    /* initialize loss */
    Loss loss;
    LossInit(&loss, layer_get_output_shape(layer), batch_size, loss_type);


    /* set up the optimizer */
    optimizer_t optimizer;
    layer_param_ref_list_t param_refs;
    optimizer_create(&optimizer, optimizer_impl, optimizer_config);
    layer_get_param_refs(layer, &param_refs);
    optimizer_add_params(optimizer, &param_refs);

    float* loss_history = NULL;
    if (reduce_lr_after != 0) {
        loss_history = (float*)calloc(reduce_lr_after, sizeof(float));
        for (size_t i = 0; i < reduce_lr_after; i++) {
            loss_history[i] = INFINITY;
        }
    }


    LOG_TRACE("Performing initial test\n");
    float test_accuracy;
    float test_loss;
    module_test(layer, test_set, batch_size, &loss, &test_accuracy, &test_loss);

    if (callback) {
        training_info_t progress_info = {
            .epoch = 0,
            .train_loss = 0.0f,
            .train_accuracy = 0.0f,
            .test_loss = test_loss,
            .test_accuracy = test_accuracy
        };
        callback(&progress_info);
    }


    LOG_TRACE("Starting training loop\n");
    for (size_t i = 0; i < num_epochs; i++) {
        LOG_TRACE("Epoch: %d\n", i + 1);

        float train_loss = 0.0f;
        float train_accuracy = 0.0f;

        tensor_t* current_inputs = NULL;
        uint8_t* current_targets = NULL;
        dataset_iteration_begin(train_set, batch_size, true, &current_inputs, &current_targets);

        while(current_inputs != NULL) {

            if (augment_pipeline != NULL) {
                augment_pipeline_forward(augment_pipeline, current_inputs, &current_inputs);
            }

            tensor_t* output;
            layer_forward(layer, LAYER_FORWARD_TRAINING, current_inputs, &output);

            /* Loss */
            train_accuracy += LossAccuracy(&loss, output, current_targets);
            train_loss += LossCompute(&loss, output, current_targets);

            /* Backward pass */
            tensor_t* gradient;
            LossBackward(&loss, output, current_targets, &gradient);
            layer_backward(layer, gradient, NULL);
            optimizer_step(optimizer);

            dataset_iteration_next(train_set, &current_inputs, &current_targets);
        }

        const size_t train_set_size = tensor_shape_get_dim(dataset_get_shape(train_set),
            TENSOR_BATCH_DIM);
        train_loss = train_loss / (float)train_set_size;
        train_accuracy = train_accuracy / (float)train_set_size;


        /* Test */
        module_test(layer, test_set, batch_size, &loss, &test_accuracy, &test_loss);

        if (callback) {
            training_info_t progress_info = {
                .epoch = i + 1,
                .train_loss = train_loss,
                .train_accuracy = train_accuracy,
                .test_loss = test_loss,
                .test_accuracy = test_accuracy
            };
            callback(&progress_info);
        }


        /* Do we want to reduce the learning rate? */
        if (loss_history != NULL) {
            bool reduce = true;
            for (size_t j = 0; j < reduce_lr_after; j++) {
                if (train_loss < loss_history[j]) {
                    reduce = false;
                    break;
                }
            }
            loss_history[i % reduce_lr_after] = train_loss;

            if (reduce) {
                LOG_INFO("no progress for %d epochs -> reducing learning rate\n",
                    reduce_lr_after);
                float lr = optimizer_get_learning_rate(optimizer);
                optimizer_set_learning_rate(optimizer, lr / 2.0f);
            }
        }
    }


    if (loss_history != NULL) {
        free(loss_history);
    }

    optimizer_destroy(optimizer);
    LossDeinit(&loss);
}