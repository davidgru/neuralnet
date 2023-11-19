
#include "ai_pooling_layer.h"

#include <malloc.h>
#include <string.h>

#define max(a, b) ((a > b) ? a : b)
#define min(a, b) ((a < b) ? a : b)

typedef struct pooling_layer_t {
    AI_Layer hdr;
    size_t kernel_width;
    void (*pooling_operation_func)(float* input, float* output, size_t input_width, size_t input_height, size_t kernel_width);
    void (*pooling_operation_backward)(float* x, float* dy, float* dx, size_t input_width, size_t input_height, size_t kernel_width);
} pooling_layer_t;


static void pooling_layer_forward(AI_Layer* layer);
static void pooling_layer_backward(AI_Layer* layer);
static void pooling_layer_deinit(AI_Layer* layer);

static void pooling_operation_average(float* x, float* y, size_t input_width, size_t input_height, size_t kernel_width);
static void pooling_operation_max(float* x, float* y, size_t input_width, size_t input_height, size_t kernel_width);
static void pooling_operation_min(float* x, float* y, size_t input_width, size_t input_height, size_t kernel_width);

static void pooling_operation_average_backward(float* x, float* dy, float* dx, size_t input_width, size_t input_height, size_t kernel_width);
static void pooling_operation_max_backward(float* x, float* dy, float* dx, size_t input_width, size_t input_height, size_t kernel_width);
static void pooling_operation_min_backward(float* x, float* dy, float* dx, size_t input_width, size_t input_height, size_t kernel_width);

void pooling_layer_init(AI_Layer** layer, void* create_info, AI_Layer* prev_layer)
{
    AI_PoolingLayerCreateInfo* _create_info = (AI_PoolingLayerCreateInfo*)create_info;

    const size_t input_width = prev_layer->output_width;
    const size_t input_height = prev_layer->output_height;
    const size_t input_channels = prev_layer->output_channels;
    const size_t kernel_size = _create_info->kernel_width;

    const size_t output_width = input_width / kernel_size;
    const size_t output_height = input_height / kernel_size;

    const size_t input_size = input_width * input_height * input_channels * prev_layer->mini_batch_size;
    const size_t output_size = output_width * output_height * input_channels * prev_layer->mini_batch_size;

    const size_t size = sizeof(pooling_layer_t) + (input_size + output_size) * sizeof(float);

    *layer = (AI_Layer*)malloc(size);

    pooling_layer_t* _layer = (pooling_layer_t*)*layer;

    _layer->hdr.input_width = input_width;
    _layer->hdr.input_height = input_height;
    _layer->hdr.input_channels = input_channels;
    _layer->hdr.output_width = output_width;
    _layer->hdr.output_height = output_height;
    _layer->hdr.output_channels = input_channels;
    _layer->hdr.mini_batch_size = prev_layer->mini_batch_size;
    _layer->hdr.forward = pooling_layer_forward;
    _layer->hdr.backward = pooling_layer_backward;
    _layer->hdr.info = NULL;
    _layer->hdr.deinit = pooling_layer_deinit;

    _layer->kernel_width = kernel_size;

    _layer->hdr.output = (float*)(_layer + 1);
    _layer->hdr.gradient = _layer->hdr.output + output_size;

    _layer->hdr.input = 0;
    _layer->hdr.prev_gradient = 0;

    switch(_create_info->pooling_operation) {
        case AI_POOLING_AVERAGE:
            _layer->pooling_operation_func = pooling_operation_average;
            _layer->pooling_operation_backward = pooling_operation_average_backward;
            break;
        case AI_POOLING_MAX:
            _layer->pooling_operation_func = pooling_operation_max;
            _layer->pooling_operation_backward = pooling_operation_max_backward;
            break;
        case AI_POOLING_MIN:
            _layer->pooling_operation_func = pooling_operation_min;
            _layer->pooling_operation_backward = pooling_operation_min_backward;
            break;
    }
}



static void pooling_layer_forward(AI_Layer* layer)
{
    pooling_layer_t* _layer = (pooling_layer_t*)layer;

    const size_t input_width = _layer->hdr.input_width;
    const size_t input_height = _layer->hdr.input_height;
    const size_t output_width = _layer->hdr.output_width;
    const size_t output_height = _layer->hdr.output_height;
    const size_t channels = _layer->hdr.input_channels;
    const size_t kernel_width = _layer->kernel_width;

    float* x = _layer->hdr.input;
    float* y = _layer->hdr.output;

    memset(y, 0, output_width * output_height * channels * _layer->hdr.mini_batch_size * sizeof(float));
    for (size_t n = 0; n < _layer->hdr.mini_batch_size; n++) {
        for (size_t i = 0; i < channels; i++) {
            float* _x = x + n * input_width * input_height * channels + i * input_width * input_height;
            float* _y = y + n * output_width * output_height * channels + i * output_width * output_height;
            _layer->pooling_operation_func(_x, _y, input_width, input_height, kernel_width);
        }
    }
}


static void pooling_layer_backward(AI_Layer* layer)
{
    pooling_layer_t* _layer = (pooling_layer_t*)layer;

    const size_t input_width = _layer->hdr.input_width;
    const size_t input_height = _layer->hdr.input_height;
    const size_t output_width = _layer->hdr.output_width;
    const size_t output_height = _layer->hdr.output_height;
    const size_t channels = _layer->hdr.input_channels;
    const size_t kernel_width = _layer->kernel_width;

    float* x = _layer->hdr.input;
    float* dx = _layer->hdr.gradient;
    float* dy = _layer->hdr.prev_gradient;

    memset(dx, 0, input_width * input_height * channels * _layer->hdr.mini_batch_size * sizeof(float));
    for (size_t n = 0; n < _layer->hdr.mini_batch_size; n++) {
        for (size_t i = 0; i < channels; i++) {
            float* _x = x + n * input_width * input_height * channels + i * input_width * input_height;
            float* _dx = dx + n * input_width * input_height * channels + i * input_width * input_height;
            float* _dy = dy + n * output_width * output_height * channels + i * output_width * output_height;
            _layer->pooling_operation_backward(_x, _dy, _dx, input_width, input_height, kernel_width);
        }
    }
}


static void pooling_layer_deinit(AI_Layer* layer)
{
    pooling_layer_t* _layer = (pooling_layer_t*)layer;
    if (_layer)
        free(_layer);
}



static void pooling_operation_average(float* x, float* y, size_t input_width, size_t input_height, size_t kernel_width)
{
    const size_t output_width = input_width / kernel_width;
    const size_t output_height = input_height / kernel_width;

    for (size_t i = 0; i < output_height; i++) {
        for (size_t j = 0; j < output_width; j++) {
            float v = 0.0f;
            for (size_t ii = 0; ii < kernel_width; ii++) {
                for (size_t jj = 0; jj < kernel_width; jj++) {
                    v += x[(kernel_width * i + ii) * input_width + (kernel_width * j + jj)];
                }
            }
            y[i * output_width + j] += v / (kernel_width * kernel_width);
        }
    }
}


static void pooling_operation_max(float* x, float* y, size_t input_width, size_t input_height, size_t kernel_width)
{
    const size_t output_width = input_width / kernel_width;
    const size_t output_height = input_height / kernel_width;

    for (size_t i = 0; i < output_height; i++) {
        for (size_t j = 0; j < output_width; j++) {
            float v = -1e12;
            for (size_t ii = 0; ii < kernel_width; ii++) {
                for (size_t jj = 0; jj < kernel_width; jj++) {
                    v = max(v, x[(kernel_width * i + ii) * input_width + (kernel_width * j + jj)]);
                }
            }
            y[i * output_width + j] += v;
        }
    }
}


static void pooling_operation_min(float* x, float* y, size_t input_width, size_t input_height, size_t kernel_width)
{
    const size_t output_width = input_width / kernel_width;
    const size_t output_height = input_height / kernel_width;

    for (size_t i = 0; i < output_height; i++) {
        for (size_t j = 0; j < output_width; j++) {
            float v = 1e12;
            for (size_t ii = 0; ii < kernel_width; ii++) {
                for (size_t jj = 0; jj < kernel_width; jj++) {
                    v = min(v, x[(kernel_width * i + ii) * input_width + (kernel_width * j + jj)]);
                }
            }
            y[i * output_width + j] += v;
        }
    }
}


static void pooling_operation_average_backward(float* x, float* dy, float* dx, size_t input_width, size_t input_height, size_t kernel_width)
{
    const size_t output_width = input_width / kernel_width;
    const size_t output_height = input_height / kernel_width;

    for (size_t i = 0; i < output_height; i++) {
        for (size_t j = 0; j < output_width; j++) {
            for (size_t ii = 0; ii < kernel_width; ii++) {
                for (size_t jj = 0; jj < kernel_width; jj++) {
                    dx[(2 * i + ii) * input_width + 2 * j + jj] += dy[i * output_width + j] / (kernel_width * kernel_width);
                }
            }
        }
    }
}

static void pooling_operation_max_backward(float* x, float* dy, float* dx, size_t input_width, size_t input_height, size_t kernel_width)
{
    const size_t output_width = input_width / kernel_width;
    const size_t output_height = input_height / kernel_width;

    for (size_t i = 0; i < output_height; i++) {
        for (size_t j = 0; j < output_width; j++) {
            // Find the maximum value and it's position in a kernel sized block
            uint32_t argmax_i = 0;
            uint32_t argmax_j = 0;
            float max = -1e12;
            for (size_t ii = 0; ii < kernel_width; ii++) {
                for (size_t jj = 0; jj < kernel_width; jj++) {
                    uint32_t _i = (2 * i + ii) * input_width + 2 * j + jj;
                    if (x[_i] > max) {
                        max = x[_i];
                        argmax_i = ii;
                        argmax_j = jj;
                        // Store 0 as gradient everywhere
                        dx[_i] += 0;
                    }
                }
            }
            // Overwrite the gradient at the correct position
            dx[(2 * i + argmax_i) * input_width + 2 * j + argmax_j] += dy[i * output_width + j];
        }
    }
}

static void pooling_operation_min_backward(float* x, float* dy, float* dx, size_t input_width, size_t input_height, size_t kernel_width)
{
    const size_t output_width = input_width / kernel_width;
    const size_t output_height = input_height / kernel_width;

    for (size_t i = 0; i < output_height; i++) {
        for (size_t j = 0; j < output_width; j++) {
            // Find the minimum value and it's position in a kernel sized block
            uint32_t argmax_i = 0;
            uint32_t argmax_j = 0;
            float max = 1e12;
            for (size_t ii = 0; ii < kernel_width; ii++) {
                for (size_t jj = 0; jj < kernel_width; jj++) {
                    uint32_t _i = (2 * i + ii) * input_width + 2 * j + jj;
                    if (x[_i] < max) {
                        max = x[_i];
                        argmax_i = ii;
                        argmax_j = jj;
                    }
                    // Store 0 as gradient everywhere
                    dx[_i] += 0;
                }
            }
            // Overwrite the gradient at the correct position
            dx[(2 * i + argmax_i) * input_width + 2 * j + argmax_j] += dy[i * output_width + j];
        }
    }

}
