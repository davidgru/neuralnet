#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "util/ai_weight_init.h"

/* The header of every layer */

typedef struct AI_Layer {
    size_t input_width;
    size_t input_height;
    size_t input_channels;
    size_t output_width;
    size_t output_height;
    size_t output_channels;
    size_t mini_batch_size;

    uint64_t is_training;

    float* input;
    float* output;
    float* gradient;
    float* prev_gradient;

    void (*forward)(struct AI_Layer* layer);
    void (*backward)(struct AI_Layer* layer);
    void (*info)(struct AI_Layer* layer);
    void (*deinit)(struct AI_Layer* layer);
} AI_Layer;

typedef enum AI_LayerKind {
    AI_INPUT_LAYER = 0,
    AI_ACTIVATION_LAYER = 1,
    AI_LINEAR_LAYER = 2,
    AI_CONVOLUTIONAL_LAYER = 3,
    AI_POOLING_LAYER = 4,
    AI_DROPOUT_LAYER = 5,
} AI_LayerKind;

/* The header of every create info */

typedef struct AI_LayerCreateInfo {
    AI_LayerKind type;
    void* create_info;
} AI_LayerCreateInfo;


typedef struct AI_InputLayerCreateInfo {
    size_t input_width;
    size_t input_height;
    size_t input_channels;
    size_t batch_size;
} AI_InputLayerCreateInfo;

/* Use to initialise an activation layer */

typedef enum AI_ActivationFunctionEnum {
    AI_ACTIVATION_FUNCTION_SIGMOID,
    AI_ACTIVATION_FUNCTION_TANH,
    AI_ACTIVATION_FUNCTION_RELU,
    AI_ACTIVATION_FUNCTION_LEAKY_RELU,
    AI_ACTIVATION_FUNCTION_SOFTMAX,
} AI_ActivationFunctionEnum;

typedef struct AI_ActivationLayerCreateInfo {
    AI_ActivationFunctionEnum activation_function;
} AI_ActivationLayerCreateInfo;


/* Use to initialise a linear layer */

typedef struct AI_LinearLayerCreateInfo {
    size_t output_size;
    float learning_rate;
    float gradient_clipping_threshold;
    AI_FCLayerWeightInit weight_init;
    AI_FCLayerBiasInit bias_init;
} AI_LinearLayerCreateInfo;


/* Use to initialise a pooling layer */

typedef enum AI_PoolingOperationEnum {
    AI_POOLING_AVERAGE,
    AI_POOLING_MAX,
    AI_POOLING_MIN,
} AI_PoolingOperationEnum;

typedef struct AI_PoolingLayerCreateInfo {
    size_t kernel_width;
    AI_PoolingOperationEnum pooling_operation;
} AI_PoolingLayerCreateInfo;


/* Use to initialise a convolutional layer */

typedef struct AI_ConvolutionalLayerCreateInfo {
    size_t output_channels;
    size_t filter_width;
    float learning_rate;
    float gradient_clipping_threshold;
    AI_ConvLayerWeightInit weight_init;
    AI_ConvLayerBiasInit bias_init;
} AI_ConvolutionalLayerCreateInfo;

/* Use to initialise a dropout layer */

typedef struct AI_DropoutLayerCreateInfo {
    float dropout_rate;
} AI_DropoutLayerCreateInfo;

void AI_LayerInit(AI_Layer** layer, AI_LayerCreateInfo* create_info, AI_Layer* prev_layer);
void AI_LayerLink(AI_Layer* layer, AI_Layer* prev_layer, AI_Layer* next_layer);
void AI_LayerForward(AI_Layer* layer);
void AI_LayerBackward(AI_Layer* layer);
void AI_LayerInfo(AI_Layer* layer);
void AI_LayerDeinit(AI_Layer* layer);
