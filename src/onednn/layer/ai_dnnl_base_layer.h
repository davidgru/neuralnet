#pragma once

#include "dnnl.h"

#include <stdint.h>
#include <stdbool.h>

typedef struct dnnl_layer_t {
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

    uint32_t(*fwd_pass_init)(struct dnnl_layer_t* layer, struct dnnl_layer_t* prev_layer);
    uint32_t(*bwd_pass_init)(struct dnnl_layer_t* layer, struct dnnl_layer_t* next_layer);
    uint32_t(*fwd)(struct dnnl_layer_t* layer);
    uint32_t(*bwd)(struct dnnl_layer_t* layer);
    uint32_t(*destroy)(struct dnnl_layer_t* layer);

} dnnl_layer_t;


typedef enum dnnl_layer_kind_t {
    dnnl_layer_kind_input = 0,
    dnnl_layer_kind_activation,
    dnnl_layer_kind_linear,
    dnnl_layer_kind_convolutional,
    dnnl_layer_kind_pooling,
    dnnl_layer_kind_dropout,
    dnnl_layer_kind_reorder,
    dnnl_layer_kind_max = 0x7FFFFFFF
} dnnl_layer_kind_t;


typedef struct dnnl_layer_create_info_t {
    dnnl_layer_kind_t layer_kind; // specifies the layer kind 
    void* layer_create_info; // the custom layer create info
} dnnl_layer_create_info_t;


typedef struct dnnl_input_layer_create_info_t {
    size_t N;
    size_t C;
    size_t H;
    size_t W;
    dnnl_engine_t engine;
    dnnl_stream_t stream;
} dnnl_input_layer_create_info_t;

typedef enum dnnl_activation_kind_t {
    dnnl_activation_kind_relu = 0,
    dnnl_activation_kind_tanh,
    dnnl_activation_kind_logistic,
    dnnl_activation_kind_max = 0x7FFFFFFF
} dnnl_activation_kind_t;


typedef struct dnnl_activation_layer_create_info_t {
    dnnl_activation_kind_t activation;
    float alpha;
    float beta;
    uint64_t allow_reorder;
} dnnl_activation_layer_create_info_t;


typedef enum dnnl_linear_layer_weight_init_kind_t {
    dnnl_linear_layer_weight_init_kind_xavier = 0,
    dnnl_linear_layer_weight_init_kind_he,
    dnnl_linear_layer_weight_init_kind_max = 0x7FFFFFFF
} dnnl_linear_layer_weight_init_kind_t;

typedef enum dnnl_linear_layer_bias_init_kind_t {
    dnnl_linear_layer_bias_init_kind_zeros = 0,
    dnnl_linear_layer_bias_init_kind_max = 0x7FFFFFFF
} dnnl_linear_layer_bias_init_kind_t;


typedef struct dnnl_linear_layer_create_info_t {
    size_t OC; // size of the output
    dnnl_linear_layer_weight_init_kind_t weight_init;
    dnnl_linear_layer_bias_init_kind_t bias_init;
    float learning_rate;
    uint32_t allow_reorder;
} dnnl_linear_layer_create_info_t;



typedef enum dnnl_convolutional_layer_weight_init_kind_t {
    dnnl_convolutional_layer_weight_init_kind_xavier = 0,
    dnnl_convolutional_layer_weight_init_kind_he,
    dnnl_convolutional_layer_weight_init_kind_max = 0x7FFFFFFF
} dnnl_convolutional_layer_weight_init_kind_t;

typedef enum dnnl_convolutional_layer_bias_init_kind_t {
    dnnl_convolutional_layer_bias_init_kind_zeros = 0,
    dnnl_convolutional_layer_bias_init_kind_max = 0x7FFFFFFF
} dnnl_convolutional_layer_bias_init_kind_t;


typedef struct dnnl_convolutional_layer_create_info_t {
    size_t OC; // output channels
    size_t KH; // kernel height
    size_t KW; // kernel width
    size_t SH; // stride in horizontal direction
    size_t SW; // stride in vertical direction
    size_t PT; // padding top
    size_t PL; // padding left
    size_t PB; // padding bottom
    size_t PR; // padding right

    dnnl_convolutional_layer_weight_init_kind_t weight_init;
    dnnl_convolutional_layer_bias_init_kind_t bias_init;

    float learning_rate;
    int32_t dummy;
} dnnl_convolutional_layer_create_info_t;

typedef enum dnnl_pooling_kind_t {
    dnnl_pooling_max = 0,
    dnnl_pooling_avg_include_padding,
    dnnl_pooling_avg,
    dnnl_pooling_maxval = 0x7FFFFFFF
} dnnl_pooling_kind_t;

typedef struct dnnl_pooling_layer_create_info_t {
    dnnl_pooling_kind_t pooling_kind;
    size_t KH;
    size_t KW;
    size_t SH;
    size_t SW;
    size_t PT;
    size_t PL;
    size_t PB;
    size_t PR;
} dnnl_pooling_layer_create_info_t;

struct _dnnl_loss_t;
typedef struct _dnnl_loss_t _dnnl_loss_t;

uint32_t dnnl_layer_create(dnnl_layer_t** layer, dnnl_layer_create_info_t* create_info);
uint32_t dnnl_layer_fwd_pass_init(dnnl_layer_t* layer, dnnl_layer_t* prev_layer);
uint32_t dnnl_layer_bwd_pass_init(dnnl_layer_t* layer, dnnl_layer_t* next_layer);
uint32_t dnnl_layer_bwd_pass_init_loss(dnnl_layer_t* layer, _dnnl_loss_t* loss);
uint32_t dnnl_layer_set_input_handle(dnnl_layer_t* layer, float* input);
float* dnnl_layer_get_output_handle(dnnl_layer_t* layer);
uint32_t dnnl_layer_fwd(dnnl_layer_t* layer);
uint32_t dnnl_layer_bwd(dnnl_layer_t* layer);
uint32_t dnnl_layer_destroy(dnnl_layer_t* layer);
