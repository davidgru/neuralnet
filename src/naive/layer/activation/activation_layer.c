#include <math.h>
#include <stdlib.h>

#include "log.h"

#include "tensor/tensor_impl.h"
#include "tensor/tensor_math.h"
#include "util/ai_math.h"
#include "layer/activation_layer.h"
#include "activation_layer_internal.h"

typedef void (*activation_func_t)(const tensor_t* in, tensor_t* out);
typedef void (*activation_bwd_func_t)(const tensor_t* in, tensor_t* out);


typedef struct activation_layer_t {
    activation_func_t activation_function;
    activation_bwd_func_t derivative;
} activation_layer_t;

/* Activation functions and derivatives */
static void sigmoid(const tensor_t* in, tensor_t* out)
{
    const size_t size = tensor_get_size(in);

    if (in->device == device_cpu) {
        sigmoid_cpu(in->data, out->data, size);
    } else if (in->device == device_gpu) {
#if defined(USE_GPU)
        sigmoid_gpu(in->data, out->data, size);
#else
        LOG_ERROR("Invalid device.\n");
#endif
    } else {
        LOG_ERROR("Invalid device.\n");
    }
}

static void _tanh(const tensor_t* in, tensor_t* out)
{
    const size_t size = tensor_get_size(in);

    if (in->device == device_cpu) {
        tanh_cpu(in->data, out->data, size);
    } else if (in->device == device_gpu) {
#if defined(USE_GPU)
        tanh_gpu(in->data, out->data, size);
#else
        LOG_ERROR("Invalid device.\n");
#endif
    } else {
        LOG_ERROR("Invalid device.\n");
    }
}

static void relu(const tensor_t* in, tensor_t* out)
{
    const size_t size = tensor_get_size(in);

    if (in->device == device_cpu) {
        relu_cpu(in->data, out->data, size);
    } else if (in->device == device_gpu) {
#if defined(USE_GPU)
        relu_gpu(in->data, out->data, size);
#else
        LOG_ERROR("Invalid device.\n");
#endif
    } else {
        LOG_ERROR("Invalid device.\n");
    }
}

static void dsigmoid(const tensor_t* in, tensor_t* out)
{
    const size_t size = tensor_get_size(in);

    if (in->device == device_cpu) {
        dsigmoid_cpu(in->data, out->data, size);
    } else if (in->device == device_gpu) {
#if defined(USE_GPU)
        dsigmoid_gpu(in->data, out->data, size);
#else
        LOG_ERROR("Invalid device.\n");
#endif
    } else {
        LOG_ERROR("Invalid device.\n");
    }
}

static void dtanh(const tensor_t* in, tensor_t* out)
{
    const size_t size = tensor_get_size(in);

    if (in->device == device_cpu) {
        dtanh_cpu(in->data, out->data, size);
    } else if (in->device == device_gpu) {
#if defined(USE_GPU)
        dtanh_gpu(in->data, out->data, size);
#else
        LOG_ERROR("Invalid device.\n");
#endif
    } else {
        LOG_ERROR("Invalid device.\n");
    }
}

static void drelu(const tensor_t* in, tensor_t* out)
{
    const size_t size = tensor_get_size(in);

    if (in->device == device_cpu) {
        drelu_cpu(in->data, out->data, size);
    } else if (in->device == device_gpu) {
#if defined(USE_GPU)
        drelu_gpu(in->data, out->data, size);
#else
        LOG_ERROR("Invalid device.\n");
#endif
    } else {
        LOG_ERROR("Invalid device.\n");
    }
}


static uint32_t activation_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape,
    device_t device
)
{
    activation_layer_t* activation_layer = (activation_layer_t*)context;
    activation_layer_create_info_t* activation_create_info =
        (activation_layer_create_info_t*)create_info;

    switch(activation_create_info->activation_function) {
        case ACTIVATION_FUNCTION_SIGMOID:
            activation_layer->activation_function = sigmoid;
            activation_layer->derivative = dsigmoid;
            break;
        case ACTIVATION_FUNCTION_TANH:
            activation_layer->activation_function = _tanh;
            activation_layer->derivative = dtanh;
            break;
        case ACTIVATION_FUNCTION_RELU:
            activation_layer->activation_function = relu;
            activation_layer->derivative = drelu;
            break;
        default:
            LOG_ERROR("Invalid activation function.\n");
            return 1;
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

    activation_layer->activation_function(input, out_output);
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

    /* Derivative */
    activation_layer->derivative(output, out_gradient);
    
    /* Chain rule */
    tensor_eltwise_mul(out_gradient, prev_gradient);
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


const layer_impl_t activation_layer_impl = {
    .init_func = activation_layer_init,
    .get_param_func = NULL, /* no params */
    .deinit_func = NULL, /* not needed */
    .forward_func = activation_layer_forward,
    .backward_func = activation_layer_backward,
    .calc_output_size = activation_layer_calc_output_shape,
    .layer_context_size = sizeof(activation_layer_t)
};
