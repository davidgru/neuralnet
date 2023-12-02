#pragma once

#include <stddef.h>

#include "core/ai_layer.h"
#include "core/ai_loss.h"
#include "core/ai_optimizer.h"

#include "dataset.h"


typedef struct ai_training_info {
    int32_t epoch;
    float train_loss;
    float train_accuracy;
    float test_loss;
    float test_accuracy;
} ai_training_info_t;


typedef void (*ai_training_callback_t)(ai_training_info_t * progress_info);


void ai_module_test(
    layer_t net,
    dataset_t test_set,
    size_t batch_size,
    AI_Loss* loss,
    float* out_accuracy,
    float* out_loss
);


void ai_module_train(
    layer_t layer,
    dataset_t train_set,
    dataset_t test_set,
    size_t num_epochs,
    size_t batch_size,
    const optimizer_impl_t* optimizer_impl,
    const optimizer_config_t* optimizer_config,
    AI_LossFunctionEnum loss_type,
    size_t reduce_lr_after,
    ai_training_callback_t callback
);
