#pragma once


#include <stddef.h>
#include <stdint.h>

#include "layer/ai_layer.h"
#include "util/ai_weight_init.h"
#include "util/ai_loss.h"

#include "layer/ai_activation_layer.h"
#include "layer/ai_convolutional_layer.h"
#include "layer/ai_dropout_layer.h"
#include "layer/ai_linear_layer.h"
#include "layer/ai_pooling_layer.h"


/* Leaving AI_LayerKind in here but remove in future. */
typedef enum AI_LayerKind {
    AI_INPUT_LAYER = 0,
    AI_ACTIVATION_LAYER = 1,
    AI_LINEAR_LAYER = 2,
    AI_CONVOLUTIONAL_LAYER = 3,
    AI_POOLING_LAYER = 4,
    AI_DROPOUT_LAYER = 5,
} AI_LayerKind;


typedef struct {
    const layer_impl_t* layer_impl;
    layer_create_info_t* create_info;
    AI_LayerKind layer_kind;
} model_desc_entry_t;


typedef struct {
    size_t num_layers;
    model_desc_entry_t* entries;
} ai_model_desc_t;


/**
 * @brief Create a model descriptor.
 * 
 * @param desc          Reference to the descriptor. Memory will be allocated.
 * @return uint32_t 
 */
uint32_t ai_model_desc_create(ai_model_desc_t** desc);


/**
 * @brief Add an activation layer.
 * 
 * @param desc                  The model descriptor.
 * @param activation_function   The type of activation function used by the layer.
 * @return uint32_t 
 */
uint32_t ai_model_desc_add_activation_layer(
    ai_model_desc_t* desc,
    activation_function_kind_t activation_function
);


/**
 * @brief Add a convolutional layer.
 * 
 * @param desc              The model descriptor.
 * @param output_channels   The amount of output channels aka number of filters.
 * @param kernel_size       The size of the square-shaped kernels.
 * @param stride            The stride in height and width dimension.
 * @param padding           The padding on top, left, bottom and right.
 * @param weight_init       The function used to initialize the weights of the layer.
 * @param bias_init         The function used to initialize the bias of the layer.
 * @return uint32_t 
 */
uint32_t ai_model_desc_add_convolutional_layer(
    ai_model_desc_t* desc,
    size_t output_channels,
    size_t kernel_size,
    size_t stride,
    size_t padding,
    AI_ConvLayerWeightInit weight_init,
    AI_ConvLayerBiasInit bias_init
);


/**
 * @brief Add a convolutional layer - extended version.
 * 
 * @param desc              The model descriptor.
 * @param output_channels   The amount of output channels aka number of filters.
 * @param kernel_height     The kernel height.
 * @param kernel_width      The kernel width.
 * @param stride_y          The stride in height dimension.
 * @param stride_x          The stride in width dimension.
 * @param padding_top       The padding on the top.
 * @param padding_left      The padding on the left.
 * @param padding_bottom    The padding on the bottom.
 * @param padding_right     The padding on the right.
 * @param weight_init       The function used to initialize the weights of the layer.
 * @param bias_init         The function used to initialize the bias of the layer.
 * @return uint32_t 
 */
uint32_t ai_model_desc_add_convolutional_layer_ext(
    ai_model_desc_t* desc,
    size_t output_channels,
    size_t kernel_height,
    size_t kernel_width,
    size_t stride_y,
    size_t stride_x,
    size_t padding_top,
    size_t padding_left,
    size_t padding_bottom,
    size_t padding_right,
    AI_ConvLayerWeightInit weight_init,
    AI_ConvLayerBiasInit bias_init
);


/**
 * @brief Add a droupout layer.
 * 
 * @param desc          The model descriptor.
 * @param dropout_rate  The dropout rate
 * @return uint32_t 
 */
uint32_t ai_model_desc_add_dropout_layer(ai_model_desc_t* desc, float dropout_rate);


/**
 * @brief Add a linear aka fully connected aka dense layer.
 * 
 * @param desc              The model descriptor.
 * @param output_size       The output size of the layer aka number of nodes.
 * @param weight_init       The function used to initialize the weights of the layer.
 * @param bias_init         The function used to initialize the bias of the layer.
 * @return uint32_t 
 */
uint32_t ai_model_desc_add_linear_layer(
    ai_model_desc_t* desc,
    size_t output_size,
    AI_FCLayerWeightInit weight_init,
    AI_FCLayerBiasInit bias_init
);


/**
 * @brief Add a pooling layer.
 * 
 * @param desc              The model descriptor.
 * @param kernel_size       The size of the square-shaped kernel.
 * @param stride            The stride in height and width dimension.
 * @param padding           The padding on top, left, bottom and right.
 * @param pooling_kind      The kind of pooling algorithm.
 * @return uint32_t 
 */
uint32_t ai_model_desc_add_pooling_layer(
    ai_model_desc_t* desc,
    size_t kernel_size,
    size_t stride,
    size_t padding,
    pooling_kind_t pooling_kind
);


/**
 * @brief Add a pooling layer - extended version.
 * 
 * @param desc              The model descriptor.
 * @param kernel_height     The height of the kernel.
 * @param kernel_width      The width of the kernel.
 * @param stride_y          The stride in height dimension.
 * @param stride_x          The stride in width dimension.
 * @param padding_top       The padding on the top.
 * @param padding_left      The padding on the left.
 * @param padding_bottom    The padding on the bottom.
 * @param padding_right     The padding on the right.
 * @param pooling_kind      The kind of pooling algorithm.
 * @return uint32_t 
 */
uint32_t ai_model_desc_add_pooling_layer_ext(
    ai_model_desc_t* desc,
    size_t kernel_height,
    size_t kernel_width,
    size_t stride_y,
    size_t stride_x,
    size_t padding_top,
    size_t padding_left,
    size_t padding_bottom,
    size_t padding_right,
    pooling_kind_t pooling_kind
);


/**
 * @brief Print a model summary to stdout.
 * 
 * @param desc The model descriptor.
 * @return uint32_t 
 */
uint32_t ai_model_desc_dump(ai_model_desc_t* desc);


/**
 * @brief Free model descriptor resources.
 * 
 * @param desc The model descriptor.
 * @return uint32_t 
 */
uint32_t ai_model_desc_destroy(ai_model_desc_t* desc);
