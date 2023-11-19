#pragma once

#include "dnnl.h"

#include <stdint.h>
#include <stdbool.h>

typedef struct ai_dnnl_layer_t {
    size_t N; // batch size
    
    size_t IC; // num input channels
    size_t IH; // input height
    size_t IW; // input width

    size_t OC; // num output channels
    size_t OH; // output height
    size_t OW; // output width

    uint64_t allow_reorder;

    dnnl_engine_t engine;
    dnnl_stream_t stream;

    dnnl_memory_t src_mem; // the input of the layer
    dnnl_memory_t dst_mem; // the output of the layer
    dnnl_memory_t diff_src_mem; // the gradients wrt. input of the layer
    dnnl_memory_t diff_dst_mem; // the gradients wrt. output of the layer

    uint32_t(*fwd_pass_init)(struct ai_dnnl_layer_t* layer, struct ai_dnnl_layer_t* prev_layer);
    uint32_t(*bwd_pass_init)(struct ai_dnnl_layer_t* layer, struct ai_dnnl_layer_t* next_layer);
    uint32_t(*fwd)(struct ai_dnnl_layer_t* layer);
    uint32_t(*bwd)(struct ai_dnnl_layer_t* layer);
    uint32_t(*destroy)(struct ai_dnnl_layer_t* layer);

} ai_dnnl_layer_t;


typedef enum ai_dnnl_layer_kind_t {
    ai_dnnl_layer_kind_input = 0,
    ai_dnnl_layer_kind_activation,
    ai_dnnl_layer_kind_linear,
    ai_dnnl_layer_kind_convolutional,
    ai_dnnl_layer_kind_pooling,
    ai_dnnl_layer_kind_dropout,
    ai_dnnl_layer_kind_reorder,
    ai_dnnl_layer_kind_max = 0x7FFFFFFF
} ai_dnnl_layer_kind_t;


typedef struct ai_dnnl_layer_create_info_t {
    ai_dnnl_layer_kind_t layer_kind; // specifies the layer kind 
    void* layer_create_info; // the custom layer create info
} ai_dnnl_layer_create_info_t;


typedef struct ai_dnnl_input_layer_create_info_t {
    size_t N;
    size_t C;
    size_t H;
    size_t W;
    dnnl_engine_t engine;
    dnnl_stream_t stream;
} ai_dnnl_input_layer_create_info_t;

typedef enum ai_dnnl_activation_kind_t {
    ai_dnnl_activation_kind_relu = 0,
    ai_dnnl_activation_kind_tanh,
    ai_dnnl_activation_kind_logistic,
    ai_dnnl_activation_kind_max = 0x7FFFFFFF
} ai_dnnl_activation_kind_t;


typedef struct ai_dnnl_activation_layer_create_info_t {
    ai_dnnl_activation_kind_t activation;
    float alpha;
    float beta;
    uint64_t allow_reorder;
} ai_dnnl_activation_layer_create_info_t;


typedef enum ai_dnnl_linear_layer_weight_init_kind_t {
    ai_dnnl_linear_layer_weight_init_kind_xavier = 0,
    ai_dnnl_linear_layer_weight_init_kind_he,
    ai_dnnl_linear_layer_weight_init_kind_max = 0x7FFFFFFF
} ai_dnnl_linear_layer_weight_init_kind_t;

typedef enum ai_dnnl_linear_layer_bias_init_kind_t {
    ai_dnnl_linear_layer_bias_init_kind_zeros = 0,
    ai_dnnl_linear_layer_bias_init_kind_max = 0x7FFFFFFF
} ai_dnnl_linear_layer_bias_init_kind_t;


typedef struct ai_dnnl_linear_layer_create_info_t {
    size_t OC; // size of the output
    ai_dnnl_linear_layer_weight_init_kind_t weight_init;
    ai_dnnl_linear_layer_bias_init_kind_t bias_init;
    float learning_rate;
    uint32_t allow_reorder;
} ai_dnnl_linear_layer_create_info_t;



typedef enum ai_dnnl_convolutional_layer_weight_init_kind_t {
    ai_dnnl_convolutional_layer_weight_init_kind_xavier = 0,
    ai_dnnl_convolutional_layer_weight_init_kind_he,
    ai_dnnl_convolutional_layer_weight_init_kind_max = 0x7FFFFFFF
} ai_dnnl_convolutional_layer_weight_init_kind_t;

typedef enum ai_dnnl_convolutional_layer_bias_init_kind_t {
    ai_dnnl_convolutional_layer_bias_init_kind_zeros = 0,
    ai_dnnl_convolutional_layer_bias_init_kind_max = 0x7FFFFFFF
} ai_dnnl_convolutional_layer_bias_init_kind_t;


typedef struct ai_dnnl_convolutional_layer_create_info_t {
    size_t OC; // output channels
    size_t KH; // kernel height
    size_t KW; // kernel width
    size_t SH; // stride in horizontal direction
    size_t SW; // stride in vertical direction
    size_t PT; // padding top
    size_t PL; // padding left
    size_t PB; // padding bottom
    size_t PR; // padding right

    ai_dnnl_convolutional_layer_weight_init_kind_t weight_init;
    ai_dnnl_convolutional_layer_bias_init_kind_t bias_init;

    float learning_rate;
    int32_t dummy;
} ai_dnnl_convolutional_layer_create_info_t;

typedef enum ai_dnnl_pooling_kind_t {
    ai_dnnl_pooling_max = 0,
    ai_dnnl_pooling_avg_include_padding,
    ai_dnnl_pooling_avg,
    ai_dnnl_pooling_maxval = 0x7FFFFFFF
} ai_dnnl_pooling_kind_t;

typedef struct ai_dnnl_pooling_layer_create_info_t {
    ai_dnnl_pooling_kind_t pooling_kind;
    size_t KH;
    size_t KW;
    size_t SH;
    size_t SW;
    size_t PT;
    size_t PL;
    size_t PB;
    size_t PR;
} ai_dnnl_pooling_layer_create_info_t;

struct _ai_dnnl_loss_t;
typedef struct _ai_dnnl_loss_t _ai_dnnl_loss_t;

uint32_t ai_dnnl_layer_create(ai_dnnl_layer_t** layer, ai_dnnl_layer_create_info_t* create_info);
uint32_t ai_dnnl_layer_fwd_pass_init(ai_dnnl_layer_t* layer, ai_dnnl_layer_t* prev_layer);
uint32_t ai_dnnl_layer_bwd_pass_init(ai_dnnl_layer_t* layer, ai_dnnl_layer_t* next_layer);
uint32_t ai_dnnl_layer_bwd_pass_init_loss(ai_dnnl_layer_t* layer, _ai_dnnl_loss_t* loss);
uint32_t ai_dnnl_layer_set_input_handle(ai_dnnl_layer_t* layer, float* input);
float* ai_dnnl_layer_get_output_handle(ai_dnnl_layer_t* layer);
uint32_t ai_dnnl_layer_fwd(ai_dnnl_layer_t* layer);
uint32_t ai_dnnl_layer_bwd(ai_dnnl_layer_t* layer);
uint32_t ai_dnnl_layer_destroy(ai_dnnl_layer_t* layer);
