#pragma once


#include <stddef.h>
#include <stdint.h>


// #include "ai_layer/ai_dnnl_base_layer.h"
// #include "util/ai_dnnl_loss.h"


typedef struct ai_input_dims_t {
    size_t N;
    size_t C;
    size_t H;
    size_t W;
} ai_input_dims_t;


typedef struct ai_model_desc_t {
    ai_input_dims_t input_shape;
    size_t num_layers;
    ai_dnnl_layer_create_info_t* create_infos;
    ai_dnnl_loss_kind_t loss;
} ai_model_desc_t;

uint32_t ai_model_desc_create(ai_model_desc_t** desc, ai_input_dims_t* input_shape, int loss);

uint32_t ai_model_desc_dump(ai_model_desc_t* desc);

uint32_t ai_model_desc_add_activation_layer(ai_model_desc_t* desc, int kind);
uint32_t ai_model_desc_add_linear_layer(ai_model_desc_t* desc, float learning_rate, size_t OC, int weight_init, ai_dnnl_linear_layer_bias_init_kind_t bias_init);
uint32_t ai_model_desc_add_convolutional_layer(ai_model_desc_t* desc, float learning_rate, size_t OC, size_t KH, size_t KW, size_t SH, size_t SW, size_t PT, size_t PL, size_t PB, size_t PR, ai_dnnl_convolutional_layer_weight_init_kind_t weight_init, ai_dnnl_convolutional_layer_bias_init_kind_t bias_init);
uint32_t ai_model_desc_add_pooling_layer(ai_model_desc_t* desc, size_t KH, size_t KW, size_t SH, size_t SW, size_t PT, size_t PL, size_t PB, size_t PR, ai_dnnl_pooling_kind_t kind);

uint32_t ai_model_desc_destroy(ai_model_desc_t* desc);
