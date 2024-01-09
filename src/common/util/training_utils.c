#include <stdbool.h>
#include <malloc.h>
#include <math.h>

#include "augment/random_crop.h"
#include "augment/image_flip.h"

#include "util/ai_math.h"

#include "log.h"

#include "training_utils.h"

size_t module_get_num_params(layer_t module)
{
    layer_param_ref_list_t param_refs;
    layer_get_param_refs(module, &param_refs);

    size_t num_params = 0;
    for (size_t i = 0; i < param_refs.num_params; i++) {
        num_params += tensor_size_from_shape(
                tensor_get_shape(param_refs.param_refs[i].param));
    }
    return num_params;
}


void module_test(
    layer_t net,
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
        layer_forward(net, LAYER_FORWARD_INFERENCE, current_inputs, &current_output);

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


void module_test_10crop(
    layer_t net,
    dataset_t test_set,
    size_t batch_size,
    size_t padding,
    Loss* loss,
    float* out_accuracy,
    float* out_loss
)
{
    if (loss != NULL) {
        LOG_ERROR("loss computation not supported at this time\n");
        return;
    }

    const tensor_shape_t* input_shape = dataset_get_shape(test_set);
    const tensor_shape_t* output_shape = layer_get_output_shape(net);

    const size_t input_channels = tensor_shape_get_dim(input_shape, TENSOR_CHANNEL_DIM);
    const size_t input_height = tensor_shape_get_dim(input_shape, TENSOR_HEIGHT_DIM);
    const size_t input_width = tensor_shape_get_dim(input_shape, TENSOR_WIDTH_DIM);
    const size_t image_size = input_channels * input_height * input_width;

    const size_t samples_per_batch = batch_size / 10;

    /* create temporary buffer to hold cropped images */
    tensor_t input_tmp;
    tensor_shape_t input_tmp_shape = make_tensor_shape(
        tensor_shape_get_depth_dim(input_shape),
        batch_size,
        input_channels,
        input_height,
        input_width
    );
    tensor_allocate(&input_tmp, &input_tmp_shape);
    tensor_fill(&input_tmp, 0.0f);
    float* input_buf = tensor_get_data(&input_tmp);

    /* create a temporary output buffer to accumulate probabilities */
    const size_t num_classes = tensor_shape_get_dim(output_shape, TENSOR_CHANNEL_DIM);
    tensor_t output_tmp;
    tensor_shape_t output_tmp_shape = make_tensor_shape(1, num_classes);
    tensor_allocate(&output_tmp, &output_tmp_shape);
    float* output_buf = tensor_get_data(&output_tmp);

    float* softmax_buf = calloc(num_classes, sizeof(float));


    tensor_t* current_inputs = NULL;
    uint8_t* current_targets = NULL;
    dataset_iteration_begin(test_set, samples_per_batch, false, &current_inputs, &current_targets);

    uint32_t iteration = 0;

    uint32_t correct_predictions = 0;

    while (current_inputs != NULL) {
        const tensor_shape_t* current_input_shape = tensor_get_shape(current_inputs);
        const float* current_input_batch_buf = tensor_get_data_const(current_inputs);

        const size_t current_batch_size = tensor_shape_get_dim(current_input_shape, TENSOR_BATCH_DIM);


        for (size_t i = 0; i < current_batch_size; i++) {
            const float* current_input_image = &current_input_batch_buf[image_size * i];
            float* current_input_buf = &input_buf[image_size * 10 * i];

            /* center crop - could also do memcpy */
            crop_image(current_input_image, current_input_buf, input_channels, input_height, input_width, 0, 0);
            crop_image(current_input_image, &current_input_buf[image_size], input_channels, input_height, input_width, 0, 0);
            flip_image_inplace(&current_input_buf[image_size], input_channels, input_height, input_width, true, false);

            /* top left crop */
            crop_image(current_input_image, &current_input_buf[image_size * 2], input_channels, input_height, input_width, -padding, -padding);
            crop_image(current_input_image, &current_input_buf[image_size * 3], input_channels, input_height, input_width, -padding, -padding);
            flip_image_inplace(&current_input_buf[image_size * 3], input_channels, input_height, input_width, true, false);

            /* top right crop */
            crop_image(current_input_image, &current_input_buf[image_size * 4], input_channels, input_height, input_width, -padding, padding);
            crop_image(current_input_image, &current_input_buf[image_size * 5], input_channels, input_height, input_width, -padding, padding);
            flip_image_inplace(&current_input_buf[image_size * 5], input_channels, input_height, input_width, true, false);


            /* bottom left crop */
            crop_image(current_input_image, &current_input_buf[image_size * 6], input_channels, input_height, input_width, padding, -padding);
            crop_image(current_input_image, &current_input_buf[image_size * 7], input_channels, input_height, input_width, padding, -padding);
            flip_image_inplace(&current_input_buf[image_size * 7], input_channels, input_height, input_width, true, false);

            /* bottom right crop */
            crop_image(current_input_buf, &current_input_buf[image_size * 8], input_channels, input_height, input_width, padding, padding);
            crop_image(current_input_buf, &current_input_buf[image_size * 9], input_channels, input_height, input_width, padding, padding);
            flip_image_inplace(&current_input_buf[image_size * 9], input_channels, input_height, input_width, true, false);
        }


        /* forward */
        tensor_t* current_output = NULL;
        layer_forward(net, LAYER_FORWARD_INFERENCE, &input_tmp, &current_output);
        const float* current_output_buf = tensor_get_data_const(current_output);

        /* accumulate softmax activations */
        for (size_t i = 0; i < current_batch_size; i++) {
            tensor_fill(&output_tmp, 0.0f);
            for (size_t j = 0; j < 10; j++) {
                softmaxv(&current_output_buf[num_classes * 10 * i + num_classes * j], softmax_buf, num_classes);
                VectorAdd(output_buf, softmax_buf, num_classes);
            }

            const uint32_t prediction = argmax(output_buf, num_classes);
            correct_predictions += prediction == (uint32_t)current_targets[i];
        }

        /* prepare next round */
        dataset_iteration_next(test_set, &current_inputs, &current_targets);

        iteration++;
    }

    const size_t dataset_size = tensor_shape_get_dim(dataset_get_shape(test_set), TENSOR_BATCH_DIM);
    *out_accuracy = (float)correct_predictions / dataset_size;
    if (out_loss != NULL) {
        *out_loss = NAN;
    }

    destroy_tensor_shape(&input_tmp_shape);
    destroy_tensor_shape(&output_tmp_shape);
    tensor_destory(&input_tmp);
    tensor_destory(&output_tmp);
    free(softmax_buf);
}


void module_train(
    layer_t layer,
    dataset_t train_set,
    augment_pipeline_t augment_pipeline,
    size_t num_epochs,
    size_t batch_size,
    const optimizer_impl_t* optimizer_impl,
    const optimizer_config_t* optimizer_config,
    learning_rate_schedule_func_t lr_schedule,
    Loss* loss,
    training_callback_t callback
)
{
    /* set up the optimizer */
    optimizer_t optimizer;
    layer_param_ref_list_t param_refs;
    optimizer_create(&optimizer, optimizer_impl, optimizer_config);
    layer_get_param_refs(layer, &param_refs);
    optimizer_add_params(optimizer, &param_refs);


    if (callback || lr_schedule) {
        const training_state_t state = {
            .model = layer,
            .optimizer = optimizer,
            .epoch = 0,
            .train_loss = INFINITY,
            .train_accuracy = 0.0f
        };
        if (lr_schedule) {
            const float lr = lr_schedule(&state);
            optimizer_set_learning_rate(optimizer, lr);
        }
        if (callback) {
            callback(&state);
        }
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
            train_accuracy += LossAccuracy(loss, output, current_targets);
            train_loss += LossCompute(loss, output, current_targets);

            /* Backward pass */
            tensor_t* gradient;
            LossBackward(loss, output, current_targets, &gradient);
            layer_backward(layer, gradient, NULL);
            optimizer_step(optimizer);

            dataset_iteration_next(train_set, &current_inputs, &current_targets);
        }

        const size_t train_set_size = tensor_shape_get_dim(dataset_get_shape(train_set),
            TENSOR_BATCH_DIM);
        train_loss = train_loss / (float)train_set_size;
        train_accuracy = train_accuracy / (float)train_set_size;

        if (callback || lr_schedule) {
            const training_state_t state = {
                .model = layer,
                .optimizer = optimizer,
                .epoch = i + 1,
                .train_loss = train_loss,
                .train_accuracy = train_accuracy
            };
            if (lr_schedule) {
                const float lr = lr_schedule(&state);
                optimizer_set_learning_rate(optimizer, lr);
            }
            if (callback) {
                callback(&state);
            }
        }

    }

    optimizer_destroy(optimizer);
}