
#include "ai_dnnl_model.h"

#include <memory.h>
#include <malloc.h>
#include <stdio.h>

#include "dnnl.h"

#include "ai_util/ai_dnnl_assert.h"

static uint32_t ai_dnnl_model_forward(ai_dnnl_model_t* model, float* input)
{
    CHECK_DNNL(ai_dnnl_layer_set_input_handle(model->layers[0], input));
    for (size_t i = 0; i < model->num_layers; i++)
        CHECK_DNNL(ai_dnnl_layer_fwd(model->layers[i]));
    return 0;
dnnl_error:
    return 1;
}

static uint32_t argmax(float* x, size_t size)
{
    uint32_t max = 0;
    for (size_t i = 1; i < size; i++)
        if (x[i] > x[max])
            max = i;
    return max;
}


uint32_t ai_dnnl_model_create(ai_dnnl_model_t** model, ai_input_dims_t* input_dims, size_t num_layers, ai_dnnl_layer_create_info_t* layer_create_infos, ai_dnnl_loss_kind_t loss_kind)
{
    // Create an engine and a stream
    dnnl_engine_t engine;
    dnnl_stream_t stream;

    dnnl_engine_create(&engine, dnnl_cpu, 0);
    dnnl_stream_create(&stream, engine, dnnl_stream_default_flags);


    *model = (ai_dnnl_model_t*)malloc(sizeof(ai_dnnl_model_t));

    ai_dnnl_model_t* m = *model;

    m->layers = (ai_dnnl_layer_t**)malloc(num_layers * sizeof(ai_dnnl_layer_t**));
    m->num_layers = num_layers;
    m->physical_input_size = input_dims->C * input_dims->H * input_dims->W;

    // Create the input layer
    ai_dnnl_input_layer_create_info_t i_info;
    i_info.N = input_dims->N;
    i_info.C = input_dims->C;
    i_info.H = input_dims->H;
    i_info.W = input_dims->W;
    i_info.engine = engine;
    i_info.stream = stream;
    ai_dnnl_layer_create_info_t input_ci;
    input_ci.layer_kind = ai_dnnl_layer_kind_input;
    input_ci.layer_create_info = &i_info;
    CHECK_DNNL(ai_dnnl_layer_create(&m->input_layer, &input_ci));

    // Create all model layers
    for (size_t i = 0; i < m->num_layers; i++)
        CHECK_DNNL(ai_dnnl_layer_create(&m->layers[i], &layer_create_infos[i]));

    // Create loss
    CHECK_DNNL(ai_dnnl_loss_create(&m->loss, loss_kind));

    // Init fwd pass for all layers
    CHECK_DNNL(ai_dnnl_layer_fwd_pass_init(m->input_layer, 0));
    CHECK_DNNL(ai_dnnl_layer_fwd_pass_init(m->layers[0], m->input_layer));
    for (size_t i = 1; i < m->num_layers; i++)
        CHECK_DNNL(ai_dnnl_layer_fwd_pass_init(m->layers[i], m->layers[i-1]));

    // Init loss
    CHECK_DNNL(ai_dnnl_loss_init(m->loss, m->layers[m->num_layers-1]));
    
    // Init bwd pass for all layers
    CHECK_DNNL(ai_dnnl_layer_bwd_pass_init_loss(m->layers[m->num_layers-1], (_ai_dnnl_loss_t*)m->loss));
    for (size_t i = m->num_layers-2; i >= 1; i--)
        CHECK_DNNL(ai_dnnl_layer_bwd_pass_init(m->layers[i], m->layers[i+1]));
    CHECK_DNNL(ai_dnnl_layer_bwd_pass_init(m->input_layer, m->layers[0]));

    ai_dnnl_layer_t* last_layer = m->layers[m->num_layers - 1];
    m->physical_output_size = last_layer->IC * last_layer->IH * last_layer->IW;


    memcpy(&m->input_shape, input_dims, sizeof(ai_input_dims_t));
    m->input_shape.N = input_dims->N;
    m->input_shape.C = input_dims->C;
    m->input_shape.H = input_dims->H;
    m->input_shape.W = input_dims->W;

    return 0;
dnnl_error:
    return 1;
}

uint32_t ai_dnnl_model_create_from_desc(ai_dnnl_model_t** model, ai_model_desc_t* desc)
{
    return ai_dnnl_model_create(model, &desc->input_shape, desc->num_layers, desc->create_infos, desc->loss);
}


uint32_t ai_dnnl_model_train(ai_dnnl_model_t* model, size_t train_set_size, float* train_data, uint8_t* train_labels, size_t test_set_size, float* test_data, uint8_t* test_labels, size_t num_epochs, ai_dnnl_train_callback_t callback)
{
    ai_dnnl_training_progress_t progress_info;
    
    float test_acc = 0.0f;
    float test_loss = 0.0f;

    float train_acc = 0.0f;
    float train_loss = 0.0f;

    // Initial test
    for (size_t i = 0; i < test_set_size; i += model->input_shape.N) {
        float* input = test_data + i * model->physical_input_size;
        uint8_t* labels = test_labels + i;
        CHECK_DNNL(ai_dnnl_model_forward(model, input));
        test_acc += ai_dnnl_loss_acc(model->loss, labels);
        test_loss += ai_dnnl_loss_loss(model->loss, labels);
    }
    test_acc = test_acc * 100.0f / test_set_size;
    test_loss = test_loss / test_set_size;

    if (callback) {
        progress_info.epoch = -1;
        progress_info.train_loss = 0.0f;
        progress_info.train_acc = 0.0f;
        progress_info.test_loss = test_loss;
        progress_info.test_acc = test_acc;
        callback(&progress_info);
    }

    // Training loop
    for (uint32_t i = 0; i < num_epochs; i++) {

        train_loss = 0.0f;
        train_acc = 0.0f;

        test_loss = 0.0f;
        test_acc = 0.0f;

        // Train one epoch
        for (size_t j = 0; j < train_set_size; j += model->input_shape.N) {
            float* input = train_data + j * model->physical_input_size;
            uint8_t* labels = train_labels + j;
            // Forward pass
            CHECK_DNNL(ai_dnnl_model_forward(model, input));
            // Evalutation
            train_acc += ai_dnnl_loss_acc(model->loss, labels);
            train_loss += ai_dnnl_loss_loss(model->loss, labels);
            // Backward pass
            CHECK_DNNL(ai_dnnl_loss_bwd(model->loss, labels));
            for (size_t k = model->num_layers - 1; k >= 1; k--)
                CHECK_DNNL(ai_dnnl_layer_bwd(model->layers[k]));
        }
        train_acc = train_acc * 100.0f / train_set_size;
        train_loss = train_loss / train_set_size;

        // Test
        for (size_t j = 0; j < test_set_size; j += model->input_shape.N) {
            float* input = test_data + j * model->physical_input_size;
            uint8_t* labels = test_labels + j;
            CHECK_DNNL(ai_dnnl_model_forward(model, input));
            test_acc += ai_dnnl_loss_acc(model->loss, labels);
            test_loss += ai_dnnl_loss_loss(model->loss, labels);
        }
        test_acc = test_acc * 100.0f / test_set_size;
        test_loss = test_loss / test_set_size;

        if (callback) {
            progress_info.epoch = i;
            progress_info.train_loss = train_loss;
            progress_info.train_acc = train_acc;
            progress_info.test_loss = test_loss;
            progress_info.test_acc = test_acc;
            callback(&progress_info);
        }
    }

    return 0;
dnnl_error:
    return 1;
}

uint32_t ai_dnnl_model_destroy(ai_dnnl_model_t* model)
{
    ai_dnnl_loss_destroy(model->loss);
    for (size_t i = 0; i < model->num_layers; i++)
        ai_dnnl_layer_destroy(model->layers[i]);
    ai_dnnl_layer_destroy(model->input_layer);

    free(model->layers);
    free(model);
    return 0;
}
