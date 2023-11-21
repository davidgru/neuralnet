#include <math.h>
#include <stdlib.h>

#include "util/ai_math.h"

#include "ai_activation_layer.h"

#include "log.h"


typedef void (*activation_function_t)(float* in, float* out, size_t size);
typedef float (*activation_derivative_t)(float f);


typedef struct activation_layer_t {
    AI_Layer hdr;
    activation_function_t activation_function;
    activation_derivative_t derivative;
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
    AI_ActivationLayerCreateInfo* activation_create_info =
        (AI_ActivationLayerCreateInfo*)create_info;

    *layer = (AI_Layer*)malloc(sizeof(activation_layer_t));
    if (*layer == NULL) {
        return 1;
    }

    activation_layer_t* activation_layer = (activation_layer_t*)*layer;


    /* fill header information */
    activation_layer->hdr.input_shape = prev_layer->output_shape;
    activation_layer->hdr.output_shape = prev_layer->output_shape; /* shape does not change*/

    /* allocate owned memory */
    tensor_allocate(&activation_layer->hdr.output, &activation_layer->hdr.output_shape);
    tensor_allocate(&activation_layer->hdr.gradient, &activation_layer->hdr.input_shape);

    /* virtual functions */
    activation_layer->hdr.forward = activation_layer_forward;
    activation_layer->hdr.backward = activation_layer_backward;
    activation_layer->hdr.info = NULL;
    activation_layer->hdr.deinit = activation_layer_deinit;


    switch(activation_create_info->activation_function) {
        case AI_ACTIVATION_FUNCTION_SIGMOID:
            activation_layer->activation_function = sigmoidv;
            activation_layer->derivative = dsigmoid;
            break;
        case AI_ACTIVATION_FUNCTION_TANH:
            activation_layer->activation_function = tanhv;
            activation_layer->derivative = dtanh;
            break;
        case AI_ACTIVATION_FUNCTION_RELU:
            activation_layer->activation_function = reluv;
            activation_layer->derivative = drelu;
            break;
        case AI_ACTIVATION_FUNCTION_LEAKY_RELU:
            activation_layer->activation_function = leaky_reluv;
            activation_layer->derivative = dleaky_relu;
            break;
        case AI_ACTIVATION_FUNCTION_SOFTMAX:
            activation_layer->activation_function = softmaxv;
            activation_layer->derivative = dsoftmax;
    }

    return 0;
}



static void activation_layer_forward(AI_Layer* layer)
{
    LOG_TRACE("activation layer forward pass\n");

    activation_layer_t* activation_layer = (activation_layer_t*)layer;    


    const tensor_shape_t* shape = tensor_get_shape(activation_layer->hdr.input);
    const size_t batch_size = shape->dims[TENSOR_BATCH_DIM];
    const size_t per_batch_size = tensor_size_from_shape(shape) / batch_size;


    float* input_data = tensor_get_data(activation_layer->hdr.input);
    float* output_data = tensor_get_data(&activation_layer->hdr.output);


    for (size_t i = 0; i < batch_size; i++) {
        float* input_batch = input_data + i * per_batch_size;
        float* output_batch = output_data + i * per_batch_size;
        activation_layer->activation_function(input_batch, output_batch, per_batch_size);
    }
}

static void activation_layer_backward(AI_Layer* layer)
{
    activation_layer_t* activation_layer = (activation_layer_t*)layer;    


    const tensor_shape_t* shape = tensor_get_shape(activation_layer->hdr.input);
    const size_t size = tensor_size_from_shape(shape);


    float* gradient_data = tensor_get_data(&activation_layer->hdr.gradient);
    float* output_data = tensor_get_data(&activation_layer->hdr.output);
    float* prev_gradient_data = tensor_get_data(activation_layer->hdr.prev_gradient);


    for (size_t i = 0; i < size; i++) {
        gradient_data[i] = activation_layer->derivative(output_data[i]);
    }
    
    /* Chain rule */
    AI_VectorMul(gradient_data, prev_gradient_data, size);
}

static void activation_layer_deinit(AI_Layer* layer)
{
    activation_layer_t* activation_layer = (activation_layer_t*)layer;
    if (activation_layer != NULL) {
        tensor_destory(&activation_layer->hdr.output);
        tensor_destory(&activation_layer->hdr.gradient);
    }
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
