#pragma once

#include <stddef.h>

#include "core/layer.h"
#include "core/loss.h"
#include "core/optimizer.h"

#include "dataset/dataset.h"
#include "augment/augment_pipeline.h"


typedef struct {
    int32_t epoch;
    float train_loss;
    float train_accuracy;
    float test_loss;
    float test_accuracy;
} training_info_t;


typedef void (*training_callback_t)(training_info_t* progress_info);


void module_test(
    layer_t net,
    dataset_t test_set,
    size_t batch_size,
    Loss* loss,
    float* out_accuracy,
    float* out_loss
);


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
);
