#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "../ai_layer/ai_dnnl_base_layer.h"


typedef float(*ai_dnnl_loss_fn_t)(float* x, size_t size, uint8_t label);
typedef void(*ai_dnnl_loss_derivative_fn_t)(float* dx, float* x, size_t size, uint8_t label);

typedef struct ai_dnnl_loss_t {

    ai_dnnl_layer_t* reorder_layer;

    float* src;
    float* diff_src;

    size_t N;
    size_t C;
    size_t H;
    size_t W;

    ai_dnnl_loss_fn_t fwd_fn;
    ai_dnnl_loss_derivative_fn_t bwd_fn;

} ai_dnnl_loss_t;

typedef enum ai_dnnl_loss_kind_t {
    ai_dnnl_loss_mse = 0,
    ai_dnnl_loss_cross_entropy,
    ai_dnnl_loss_max = 0x7FFFFFFF,
} ai_dnnl_loss_kind_t;

uint32_t ai_dnnl_loss_create(ai_dnnl_loss_t** loss, ai_dnnl_loss_kind_t loss_kind);

uint32_t ai_dnnl_loss_init(ai_dnnl_loss_t* loss, ai_dnnl_layer_t* prev_layer);

uint32_t ai_dnnl_loss_acc(ai_dnnl_loss_t* loss, uint8_t* labels);
float ai_dnnl_loss_loss(ai_dnnl_loss_t* loss, uint8_t* labels);
uint32_t ai_dnnl_loss_bwd(ai_dnnl_loss_t* loss, uint8_t* labels);

uint32_t ai_dnnl_loss_destroy(ai_dnnl_loss_t* loss);
