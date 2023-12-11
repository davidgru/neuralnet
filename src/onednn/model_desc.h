#pragma once


#include <stddef.h>
#include <stdint.h>


#include "layer/dnnl_base_layer.h"
#include "util/dnnl_loss.h"


typedef struct input_dims_t {
    size_t N;
    size_t C;
    size_t H;
    size_t W;
} input_dims_t;


typedef struct model_desc_t {
    input_dims_t input_shape;
    size_t num_layers;
    dnnl_layer_create_info_t* create_infos;
    dnnl_loss_kind_t loss;
} model_desc_t;

uint32_t model_desc_create(model_desc_t** desc, input_dims_t* input_shape, dnnl_loss_kind_t loss);

uint32_t model_desc_dump(model_desc_t* desc);

uint32_t model_desc_add_activation_layer(model_desc_t* desc, dnnl_activation_kind_t kind);
uint32_t model_desc_add_linear_layer(model_desc_t* desc, float learning_rate, size_t OC, dnnl_linear_layer_weight_init_kind_t weight_init, dnnl_linear_layer_bias_init_kind_t bias_init);
uint32_t model_desc_add_convolutional_layer(model_desc_t* desc, float learning_rate, size_t OC, size_t KH, size_t KW, size_t SH, size_t SW, size_t PT, size_t PL, size_t PB, size_t PR, dnnl_convolutional_layer_weight_init_kind_t weight_init, dnnl_convolutional_layer_bias_init_kind_t bias_init);
uint32_t model_desc_add_pooling_layer(model_desc_t* desc, size_t KH, size_t KW, size_t SH, size_t SW, size_t PT, size_t PL, size_t PB, size_t PR, dnnl_pooling_kind_t kind);

uint32_t model_desc_destroy(model_desc_t* desc);
