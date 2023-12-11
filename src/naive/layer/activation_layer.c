#include <math.h>
#include <stdlib.h>

#include "util/ai_math.h"

#include "layer/activation_layer.h"

#include "log.h"


typedef void (*activation_function_t)(const float* in, float* out, size_t size);
typedef float (*activation_derivative_t)(float f);


typedef struct activation_layer_t {
    activation_function_t activation_function;
    activation_derivative_t derivative;
} activation_layer_t;


static uint32_t activation_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
);

static uint32_t activation_layer_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t* out_output
);

static uint32_t activation_layer_backward(
    layer_context_t* context,
    const tensor_t* input,
    const tensor_t* output,
    const tensor_t* prev_gradient,
    tensor_t* out_gradient
);

static uint32_t activation_layer_calc_output_shape(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
);


const layer_impl_t activation_layer_impl = {
    .init_func = activation_layer_init,
    .get_param_func = NULL, /* no params */
    .deinit_func = NULL, /* not needed */
    .forward_func = activation_layer_forward,
    .backward_func = activation_layer_backward,
    .calc_output_size = activation_layer_calc_output_shape,
    .layer_context_size = sizeof(activation_layer_t)
};


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

static void sigmoidv(const float* in, float* out, size_t size);
static void tanhv(const float* in, float* out, size_t size);
static void reluv(const float* in, float* out, size_t size);
static void leaky_reluv(const float* in, float* out, size_t size);
static void softmaxv(const float* in, float* out, size_t size);

static void dsigmoidv(const float* in, float* out, size_t size);
static void dtanhv(const float* in, float* out, size_t size);
static void dreluv(const float* in, float* out, size_t size);
static void dleaky_reluv(const float* in, float* out, size_t size);
static void dsoftmaxv(const float* in, float* out, size_t size);


static uint32_t activation_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
)
{
    activation_layer_t* activation_layer = (activation_layer_t*)context;
    activation_layer_create_info_t* activation_create_info =
        (activation_layer_create_info_t*)create_info;

    switch(activation_create_info->activation_function) {
        case ACTIVATION_FUNCTION_SIGMOID:
            activation_layer->activation_function = sigmoidv;
            activation_layer->derivative = dsigmoid;
            break;
        case ACTIVATION_FUNCTION_TANH:
            activation_layer->activation_function = tanhv;
            activation_layer->derivative = dtanh;
            break;
        case ACTIVATION_FUNCTION_RELU:
            activation_layer->activation_function = reluv;
            activation_layer->derivative = drelu;
            break;
        case ACTIVATION_FUNCTION_LEAKY_RELU:
            activation_layer->activation_function = leaky_reluv;
            activation_layer->derivative = dleaky_relu;
            break;
    }

    return 0;
}


static uint32_t activation_layer_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t* out_output
)
{
    activation_layer_t* activation_layer = (activation_layer_t*)context;    


    const tensor_shape_t* shape = tensor_get_shape(input);
    const size_t batch_size = shape->dims[TENSOR_BATCH_DIM];
    const size_t per_batch_size = tensor_size_from_shape(shape) / batch_size;


    const float* input_data = tensor_get_data_const(input);
    float* output_data = tensor_get_data(out_output);


    for (size_t i = 0; i < batch_size; i++) {
        const float* input_batch = input_data + i * per_batch_size;
        float* output_batch = output_data + i * per_batch_size;
        activation_layer->activation_function(input_batch, output_batch, per_batch_size);
    }
}


static uint32_t activation_layer_backward(
    layer_context_t* context,
    const tensor_t* input,
    const tensor_t* output,
    const tensor_t* prev_gradient,
    tensor_t* out_gradient
)
{
    activation_layer_t* activation_layer = (activation_layer_t*)context;


    const tensor_shape_t* shape = tensor_get_shape(out_gradient);
    const size_t size = tensor_size_from_shape(shape);


    const float* output_data = tensor_get_data_const(output);
    const float* prev_gradient_data = tensor_get_data_const(prev_gradient);
    float* gradient_data = tensor_get_data(out_gradient);


    for (size_t i = 0; i < size; i++) {
        gradient_data[i] = activation_layer->derivative(output_data[i]);
    }
    
    /* Chain rule */
    VectorMul(gradient_data, prev_gradient_data, size);
}


static uint32_t activation_layer_calc_output_shape(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
)
{
    /* activation layer does not change the shape */
    *out_output_shape = *input_shape;

    return 0;
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
    if (f <= 0.0)
        return 0.0f;
    else
        return 1.0f;
}

float dleaky_relu(float f)
{
    if (f <= 0)
        return 0.01f;
    else
        return 1.0f;    
}

float dsoftmax(float f)
{
    return (1.0f - f) * f;
}


void sigmoidv(const float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++)
        out[i] = sigmoid(in[i]);
}

void tanhv(const float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++)
        out[i] = tanh(in[i]);
}

void reluv(const float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++)
        out[i] = relu(in[i]);
}

void leaky_reluv(const float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++)
        out[i] = leaky_relu(in[i]);
}

void softmaxv(const float* in, float* out, size_t size)
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


void dsigmoidv(const float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++)
        out[i] = dsigmoid(in[i]);
}


void dtanhv(const float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++)
        out[i] = dtanh(in[i]);
}


void dreluv(const float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++)
        out[i] = drelu(in[i]);
}


void dleaky_reluv(const float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++)
        out[i] = dleaky_relu(in[i]);
}

void dsoftmaxv(const float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++)
        out[i] = (1.0f - in[i]) * in[i];
}
