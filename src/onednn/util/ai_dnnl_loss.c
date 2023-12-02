
#include "dnnl_loss.h"


#include "dnnl_util.h"

#include <malloc.h>
#include <math.h>
#include <stdio.h>

static float loss_mse(float* x, size_t size, uint8_t label);
static void loss_mse_bwd(float* dx, float* x, size_t size, uint8_t label);
static float loss_cross_entropy(float* x, size_t size, uint8_t label);
static void loss_cross_entropy_bwd(float* dx, float* x, size_t size, uint8_t label);


static uint32_t argmax(float* x, size_t size);

static dnnl_loss_fn_t get_loss_fn(dnnl_loss_kind_t loss_kind)
{
    static dnnl_loss_fn_t t[] = {
        loss_mse,
        loss_cross_entropy
    };
    return t[loss_kind];
}

static dnnl_loss_derivative_fn_t get_loss_derivative_fn(dnnl_loss_kind_t loss_kind)
{
    static dnnl_loss_derivative_fn_t t[] = {
        loss_mse_bwd,
        loss_cross_entropy_bwd
    };
    return t[loss_kind];
}

uint32_t dnnl_loss_create(dnnl_loss_t** loss, dnnl_loss_kind_t loss_kind)
{
    *loss = (dnnl_loss_t*)malloc(sizeof(dnnl_loss_t));

    (*loss)->fwd_fn = get_loss_fn(loss_kind);
    (*loss)->bwd_fn = get_loss_derivative_fn(loss_kind);
    
    // Create a reorder layer
    dnnl_layer_create_info_t reorder_create_info;
    reorder_create_info.layer_kind = dnnl_layer_kind_reorder;
    reorder_create_info.layer_create_info = 0;

    uint32_t status = dnnl_layer_create(&(*loss)->reorder_layer, &reorder_create_info);

    return status;
}

uint32_t dnnl_loss_init(dnnl_loss_t* loss, dnnl_layer_t* prev_layer)
{
    loss->N = prev_layer->N;
    loss->C = prev_layer->OC;
    loss->H = prev_layer->OH;
    loss->W = prev_layer->OW;

    dnnl_layer_fwd_pass_init(loss->reorder_layer, prev_layer);
    dnnl_layer_bwd_pass_init(loss->reorder_layer, 0);

    loss->src = dnnl_memory_get_data_handle(loss->reorder_layer->dst_mem);
    loss->diff_src = dnnl_memory_get_data_handle(loss->reorder_layer->diff_dst_mem);

    return 0;
}

uint32_t dnnl_loss_acc(dnnl_loss_t* loss, uint8_t* labels)
{
    uint32_t acc = 0;
    const size_t strideN = loss->C * loss->H * loss->W;
    for (size_t n = 0; n < loss->N; n++)
        acc += argmax(loss->src + n * strideN, strideN) == labels[n];
    return acc;
}

float dnnl_loss_loss(dnnl_loss_t* loss, uint8_t* labels)
{
    float l = 0.0f;
    const size_t strideN = loss->C * loss->H * loss->W;
    for (size_t n = 0; n < loss->N; n++)
        l += loss->fwd_fn(loss->src + n * strideN, strideN, labels[n]);
    return l;
}

uint32_t dnnl_loss_bwd(dnnl_loss_t* loss, uint8_t* labels)
{
    const size_t strideN = loss->C * loss->H * loss->W;
    for (size_t n = 0; n < loss->N; n++)
        loss->bwd_fn(loss->diff_src + n * strideN, loss->src + n * strideN, strideN, labels[n]);
    return 0;
}

uint32_t dnnl_loss_destroy(dnnl_loss_t* loss)
{
    uint32_t status = dnnl_layer_destroy(loss->reorder_layer);

    free(loss);

    return status;
}




static float loss_mse(float* x, size_t size, uint8_t label)
{
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        float t = (label == i) - x[i];
        sum += t * t;
    }
    return sum / size;
}


static void loss_mse_bwd(float* dx, float* x, size_t size, uint8_t label)
{
    for (size_t i = 0; i < size; i++)
        dx[i] = x[i] - (label == i);
}


static float loss_cross_entropy(float* x, size_t size, uint8_t label)
{
    return -logf(fmax(1e-12, x[label]));
}


static void loss_cross_entropy_bwd(float* dx, float* x, size_t size, uint8_t label)
{
    for (size_t i = 0; i < size; i++)
        dx[i] = x[i] - (label == i);
}



static uint32_t argmax(float* x, size_t size)
{
    uint32_t max = 0;
    for (size_t i = 1; i < size; i++)
        if (x[i] > x[max])
            max = i;
    return max;
}
