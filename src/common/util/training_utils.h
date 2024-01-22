#pragma once

#include <stddef.h>

#include "core/layer.h"
#include "core/loss.h"
#include "core/optimizer.h"

#include "dataset/dataset.h"
#include "augment/augment_pipeline.h"


typedef struct {
    layer_t model;
    optimizer_t optimizer;
    Loss* loss;
    int32_t epoch;
    float train_loss;
    float train_accuracy;
} training_state_t;



typedef void (*training_callback_t)(const training_state_t* training_state);
typedef float (*learning_rate_schedule_func_t)(const training_state_t* training_state);


size_t module_get_num_params(layer_t module);


void module_test(
    layer_t net,
    dataset_t test_set,
    size_t batch_size,
    Loss* loss,
    float* out_accuracy,
    float* out_loss
);


void module_test_10crop(
    layer_t net,
    dataset_t test_set,
    size_t batch_size,
    size_t padding,
    Loss* loss,
    float* out_accuracy,
    float* out_loss
);


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
);
