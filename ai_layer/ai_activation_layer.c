
#include "ai_activation_layer.h"

#include "../ai_util/ai_math.h"

#include <math.h>
#include <stdlib.h>

typedef struct activation_layer_t {
    AI_Layer hdr;
    void(*activation_function)(float* in, float* out, size_t size);
    float(*derivative)(float f);
} activation_layer_t;

/* Activation functions and derivatives */

static float sigmoid(float f);
//static float tanh(float f);
#define tanh tanhf
static float relu(float f);
static float leaky_relu(float f);

static float dsigmoid(float f);
static float dtanh(float f);
static float drelu(float f);
static float dleaky_relu(float f);
static float dsoftmax(float f);

static void sigmoidv(float* in, float* out, size_t size);
static void tanhv(float* in, float* out, size_t size);
static void reluv(float* in, float* out, size_t size);
static void leaky_reluv(float* in, float* out, size_t size);
static void softmaxv(float* in, float* out, size_t size);

static void dsigmoidv(float* in, float* out, size_t size);
static void dtanhv(float* in, float* out, size_t size);
static void dreluv(float* in, float* out, size_t size);
static void dleaky_reluv(float* in, float* out, size_t size);
static void dsoftmaxv(float* in, float* out, size_t size);


static void activation_layer_forward(AI_Layer* layer);
static void activation_layer_backward(AI_Layer* layer);
static void activation_layer_deinit(AI_Layer* layer);


uint32_t activation_layer_init(AI_Layer** layer, void* create_info, AI_Layer* prev_layer)
{
    AI_ActivationLayerCreateInfo* _create_info = (AI_ActivationLayerCreateInfo*)create_info;

    // Allocate memory for the layer
    const size_t input_size = prev_layer->output_width * prev_layer->output_height * prev_layer->output_channels; 

    size_t size = sizeof(activation_layer_t) + 2 * input_size * prev_layer->mini_batch_size * sizeof(float);
    *layer = (AI_Layer*)malloc(size);

    activation_layer_t* _layer = (activation_layer_t*)*layer;

    _layer->hdr.input_width = prev_layer->output_width;
    _layer->hdr.input_height = prev_layer->output_height;
    _layer->hdr.input_channels = prev_layer->output_channels;
    _layer->hdr.output_width = _layer->hdr.input_width;
    _layer->hdr.output_height = _layer->hdr.input_height;
    _layer->hdr.output_channels = _layer->hdr.input_channels;
    _layer->hdr.mini_batch_size = prev_layer->mini_batch_size;

    _layer->hdr.forward = activation_layer_forward;
    _layer->hdr.backward = activation_layer_backward;
    _layer->hdr.deinit = activation_layer_deinit;

    _layer->hdr.output = (float*)(_layer + 1);
    _layer->hdr.gradient = _layer->hdr.output + input_size * _layer->hdr.mini_batch_size;

    switch(_create_info->activation_function) {
        case AI_ACTIVATION_FUNCTION_SIGMOID:
            _layer->activation_function = sigmoidv;
            _layer->derivative = dsigmoid;
            break;
        case AI_ACTIVATION_FUNCTION_TANH:
            _layer->activation_function = tanhv;
            _layer->derivative = dtanh;
            break;
        case AI_ACTIVATION_FUNCTION_RELU:
            _layer->activation_function = reluv;
            _layer->derivative = drelu;
            break;
        case AI_ACTIVATION_FUNCTION_LEAKY_RELU:
            _layer->activation_function = leaky_reluv;
            _layer->derivative = dleaky_relu;
            break;
        case AI_ACTIVATION_FUNCTION_SOFTMAX:
            _layer->activation_function = softmaxv;
            _layer->derivative = dsoftmax;
    }
}



static void activation_layer_forward(AI_Layer* layer)
{
    activation_layer_t* _layer = (activation_layer_t*)layer;

    const size_t input_size = _layer->hdr.input_width * _layer->hdr.input_height * _layer->hdr.input_channels;

    for (size_t i = 0; i < _layer->hdr.mini_batch_size; i++)
        _layer->activation_function(_layer->hdr.input + i * input_size, _layer->hdr.output + i * input_size, input_size);
}

static void activation_layer_backward(AI_Layer* layer)
{
    activation_layer_t* _layer = (activation_layer_t*)layer;

    const size_t input_size = _layer->hdr.input_width * _layer->hdr.input_height * _layer->hdr.input_channels;
    const size_t mini_batch_size = _layer->hdr.mini_batch_size;

    for (size_t i = 0; i < mini_batch_size * input_size; i++)
        _layer->hdr.gradient[i] = _layer->derivative(_layer->hdr.output[i]);
    AI_VectorMulAVX(_layer->hdr.gradient, _layer->hdr.prev_gradient, input_size * mini_batch_size);
}

static void activation_layer_deinit(AI_Layer* layer)
{
    activation_layer_t* _layer = (activation_layer_t*)layer;

    if (_layer)
        free(_layer);
}


/* Activation functions and derivatives */

float sigmoid(float f)
{
    return 1.0f / (1.0f + exp(-f));
}


// tanh is already declared in <math.h>
// float tanh(float f)
// {
//     return tanh(f);
// }

float relu(float f)
{
    if (f < 0.0f)
        return 0.0f;
    else
        return f;
}

float leaky_relu(float f)
{
    if (f < 0)
        return 0.01 * f;
    else
        return f;
}


float dsigmoid(float f)
{
    return (1.0f - f) * f;
}

float dtanh(float f)
{
    return 1 - f * f;
}

float drelu(float f)
{
    if (f < 0)
        return 0.0f;
    else
        return 1.0f;
}

float dleaky_relu(float f)
{
    if (f < 0)
        return 0.01f;
    else
        return 1.0f;    
}

float dsoftmax(float f)
{
    return (1.0f - f) * f;
}


void sigmoidv(float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++)
        out[i] = sigmoid(in[i]);
}

void tanhv(float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++)
        out[i] = tanh(in[i]);
}

void reluv(float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++)
        out[i] = relu(in[i]);
}

void leaky_reluv(float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++)
        out[i] = leaky_relu(in[i]);
}

void softmaxv(float* in, float* out, size_t size)
{
    // Find the max value
    float max = -3.4028235e38;
    for (size_t i = 0; i < size; i++)
        if (in[i] > max)
            max = in[i];
    // exp(v - max) all values
    for (size_t i = 0; i < size; i++)
        out[i] = exp(in[i] - max);
    // sum them
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++)
        sum += out[i];
    // divide every value by the sum
    sum = 1.0f / sum;
    for (size_t i = 0; i < size; i++)
        out[i] = out[i] * sum;
}


void dsigmoidv(float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++)
        out[i] = dsigmoid(in[i]);
}


void dtanhv(float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++)
        out[i] = dtanh(in[i]);
}


void dreluv(float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++)
        out[i] = drelu(in[i]);
}


void dleaky_reluv(float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++)
        out[i] = dleaky_relu(in[i]);
}

void dsoftmaxv(float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++)
        out[i] = (1.0f - in[i]) * in[i];
}
