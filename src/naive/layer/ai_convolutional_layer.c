
#include "ai_convolutional_layer.h"

#include <stdlib.h>
#include <string.h>

#include "util/ai_math.h"
#include "util/ai_gradient_clipping.h"

typedef struct convolutional_layer_t {
    AI_Layer hdr;
    size_t filter_width;
    float* w;
    float* b;
    float* dw;
    float learning_rate;
    float gradient_clipping_threshold;
} convolutional_layer_t;


static void conv_layer_forward(AI_Layer* layer);
static void conv_layer_backward(AI_Layer* layer);
static void conv_layer_deinit(AI_Layer* layer);


uint32_t convolutional_layer_init(AI_Layer** layer, void* create_info, AI_Layer* prev_layer)
{
    AI_ConvolutionalLayerCreateInfo* _create_info = (AI_ConvolutionalLayerCreateInfo*)create_info;

    const size_t input_width = prev_layer->output_width;
    const size_t input_height = prev_layer->output_height;
    const size_t input_channels = prev_layer->output_channels;
    const size_t filter_width = _create_info->filter_width;

    const size_t output_width = input_width - filter_width + 1;
    const size_t output_height = input_height - filter_width + 1;
    const size_t output_channels = _create_info->output_channels;

    const size_t weight_size = filter_width * filter_width * input_channels * output_channels;
    const size_t bias_size = output_channels;
    const size_t output_size = output_width * output_height * output_channels * prev_layer->mini_batch_size;
    const size_t gradient_size = input_width * input_height * input_channels * prev_layer->mini_batch_size;

    const size_t size = sizeof(convolutional_layer_t) + (weight_size + bias_size + output_size + gradient_size + weight_size) * sizeof(float);
    *layer = (AI_Layer*)malloc(size);

    convolutional_layer_t* _layer = (convolutional_layer_t*)*layer;

    _layer->hdr.input_width = input_width;
    _layer->hdr.input_height = input_height;
    _layer->hdr.input_channels = input_channels;
    _layer->hdr.output_width = output_width;
    _layer->hdr.output_height = output_height;
    _layer->hdr.output_channels = output_channels;
    _layer->hdr.mini_batch_size = prev_layer->mini_batch_size;
    _layer->hdr.forward = conv_layer_forward;
    _layer->hdr.backward = conv_layer_backward;
    _layer->hdr.deinit = conv_layer_deinit;

    _layer->filter_width = filter_width;
    _layer->learning_rate = _create_info->learning_rate;
    _layer->gradient_clipping_threshold = _create_info->gradient_clipping_threshold;

    _layer->w = (float*)(_layer + 1);
    _layer->b = _layer->w + weight_size;
    _layer->hdr.output = _layer->b + bias_size;
    _layer->hdr.gradient = _layer->hdr.output + output_size;
    _layer->dw = _layer->hdr.gradient + gradient_size;

    for (size_t i = 0; i < weight_size; i++)
        _layer->w[i] = _create_info->weight_init(input_width, input_height, input_channels);
    for (size_t i = 0; i < bias_size; i++)
        _layer->b[i] = _create_info->bias_init(input_width, input_height, input_channels);
}


static void conv_layer_forward(AI_Layer* layer)
{
    convolutional_layer_t* _layer = (convolutional_layer_t*)layer;

    const size_t filter_width = _layer->filter_width;
    const size_t input_width = _layer->hdr.input_width;
    const size_t input_height = _layer->hdr.input_height;
    const size_t input_channels = _layer->hdr.input_channels;
    const size_t output_width = _layer->hdr.output_width;
    const size_t output_height = _layer->hdr.output_height;
    const size_t output_channels = _layer->hdr.output_channels;
    const size_t batch_size = _layer->hdr.mini_batch_size;

    const size_t filter_size = filter_width * filter_width * input_channels;
    const size_t output_size = output_width * output_height;

    float* x = _layer->hdr.input;
    float* y = _layer->hdr.output;
    float* w = _layer->w;
    float* b = _layer->b;

    memset(y, 0, output_size * output_channels * batch_size * sizeof(float));
    for (size_t n = 0; n < batch_size; n++) {
        for (size_t i = 0; i < output_channels; i++) {
            float* _y = y + n * output_size * output_channels + i * output_size;
            for (size_t j = 0; j < input_channels; j++) {
                float* _x = x + n * input_width * input_height * input_channels + j * input_width * input_height;
                float* _w = w + i * filter_size + j * filter_width * filter_width;
                // Do a convolution with the one input channel and one filter channel to produce part of one output feature map.
                AI_MatrixConvolution(_x, _w, _y, input_width, input_height, filter_width, filter_width, 1, 1);
            }
            // Add the bias to every element of the feature map
            AI_VectorAddScalar(y, b[i], output_size);
        }
    }
}


static void conv_layer_backward(AI_Layer* layer)
{
    convolutional_layer_t* _layer = (convolutional_layer_t*)layer;

    const size_t filter_width = _layer->filter_width;
    const size_t input_width = _layer->hdr.input_width;
    const size_t input_height = _layer->hdr.input_height;
    const size_t input_channels = _layer->hdr.input_channels;
    const size_t output_width = _layer->hdr.output_width;
    const size_t output_height = _layer->hdr.output_height;
    const size_t output_channels = _layer->hdr.output_channels;
    const size_t batch_size = _layer->hdr.mini_batch_size;

    const size_t filter_size = filter_width * filter_width * input_channels;
    const size_t output_size = output_width * output_height;
    const size_t input_size = input_width * input_height;

    const float learning_rate = _layer->learning_rate;

    float* x = _layer->hdr.input;
    float* y = _layer->hdr.output;
    float* w = _layer->w;
    float* b = _layer->b;
    float* dx = _layer->hdr.gradient;
    float* dy = _layer->hdr.prev_gradient;
    float* dw = _layer->dw;

    // Calculate gradients with respect to the input and store in dx
    memset(dx, 0, input_size * input_channels * batch_size * sizeof(float));
    for (size_t n = 0; n < batch_size; n++) {
        for (size_t i = 0; i < input_channels; i++) {
            float* _dx = dx + n * input_size * input_channels + i * input_size;
            for (size_t j = 0; j < output_channels; j++) {
                float* _dy = dy + n * output_size * output_channels + j * output_size;
                float* _w = w + j + filter_size + i * filter_width * filter_width;
                AI_MatrixConvolutionPaddedRotateFilter(_dy, _w, _dx, output_width, output_height, filter_width, filter_width, 1, 1, filter_width - 1, filter_width - 1, 0, 0);
            }
        }
    }

    // Adjust filter weights
    memset(dw, 0, filter_size * output_channels * sizeof(float));
    for (size_t n = 0; n < batch_size; n++) {
        for (size_t i = 0; i < input_channels; i++) {
            float* _x = x + n * input_size * input_channels + i * input_size;
            for (size_t j = 0; j < output_channels; j++) {
                float* _dy = dy + n * output_size * output_channels + j * output_size;
                float* _dw = dw + j * filter_size + i * filter_width * filter_width;
                AI_MatrixConvolutionPadded(_x, _dy, _dw, input_width, input_height, output_width, output_height, 1, 1, 0, 0, 0, 0);
            }
        }
    }
    AI_VectorScale(dw, learning_rate, filter_size * output_channels);
    AI_ClipGradient(dw, filter_size * output_channels, _layer->gradient_clipping_threshold);
    AI_VectorSub(w, dw, filter_size * output_channels);

    // Adjust output channel bias
    for (size_t i = 0; i < output_channels; i++) {
        float _db = AI_Sum(dy + i * output_size, output_size);
        AI_ClipGradient(&_db, 1, _layer->gradient_clipping_threshold);
        b[i] -= learning_rate * _db;
    }
}


static void conv_layer_deinit(AI_Layer* layer)
{
    convolutional_layer_t* _layer = (convolutional_layer_t*)layer;

    if (_layer)
        free(_layer);
}
