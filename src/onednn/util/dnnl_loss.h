#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "../layer/dnnl_base_layer.h"


typedef float(*dnnl_loss_fn_t)(float* x, size_t size, uint8_t label);
typedef void(*dnnl_loss_derivative_fn_t)(float* dx, float* x, size_t size, uint8_t label);

typedef struct dnnl_loss_t {

    dnnl_layer_t* reorder_layer;

    float* src;
    float* diff_src;

    size_t N;
    size_t C;
    size_t H;
    size_t W;

    dnnl_loss_fn_t fwd_fn;
    dnnl_loss_derivative_fn_t bwd_fn;

} dnnl_loss_t;

typedef enum dnnl_loss_kind_t {
    dnnl_loss_mse = 0,
    dnnl_loss_cross_entropy,
    dnnl_loss_max = 0x7FFFFFFF,
} dnnl_loss_kind_t;

uint32_t dnnl_loss_create(dnnl_loss_t** loss, dnnl_loss_kind_t loss_kind);

uint32_t dnnl_loss_init(dnnl_loss_t* loss, dnnl_layer_t* prev_layer);

uint32_t dnnl_loss_acc(dnnl_loss_t* loss, uint8_t* labels);
float dnnl_loss_loss(dnnl_loss_t* loss, uint8_t* labels);
uint32_t dnnl_loss_bwd(dnnl_loss_t* loss, uint8_t* labels);

uint32_t dnnl_loss_destroy(dnnl_loss_t* loss);
