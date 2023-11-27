#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "util/ai_weight_init.h"

#include "tensor.h"
#include "tensor_impl.h"

/* The header of every layer */

typedef struct AI_Layer {
    /* Input and output shape */
    tensor_shape_t input_shape;
    tensor_shape_t output_shape;

    /* Supplied by adjacent layers */
    tensor_t* input;
    tensor_t* prev_gradient;
    
    /* Allocated and owned by the layer */
    tensor_t gradient;
    tensor_t output;

    /* Virtual functions implemented by each layer */
    void (*forward)(struct AI_Layer* layer);
    void (*backward)(struct AI_Layer* layer);
    void (*info)(struct AI_Layer* layer);
    void (*deinit)(struct AI_Layer* layer);

    uint64_t is_training;
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
    tensor_shape_t input_shape;
} AI_InputLayerCreateInfo;

/* Use to initialise an activation layer */

typedef enum AI_ActivationFunctionEnum {
    AI_ACTIVATION_FUNCTION_SIGMOID,
    AI_ACTIVATION_FUNCTION_TANH,
    AI_ACTIVATION_FUNCTION_RELU,
    AI_ACTIVATION_FUNCTION_LEAKY_RELU,
    /* AI_ACTIVATION_FUNCTION_SOFTMAX, */
} AI_ActivationFunctionEnum;

typedef struct AI_ActivationLayerCreateInfo {
    AI_ActivationFunctionEnum activation_function;
} AI_ActivationLayerCreateInfo;


/* Use to initialise a linear layer */

typedef struct AI_LinearLayerCreateInfo {
    size_t output_size;
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


/* Use to initialise a dropout layer */

typedef struct AI_DropoutLayerCreateInfo {
    float dropout_rate;
} AI_DropoutLayerCreateInfo;
