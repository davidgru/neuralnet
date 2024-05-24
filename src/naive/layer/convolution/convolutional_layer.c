#include <stdlib.h>
#include <string.h>

#include "tensor/tensor_impl.h"
#include "tensor/tensor_math.h"

#include "log.h"

#include "layer/convolutional_layer.h"
#include "convolutional_layer_internal.h"

#if defined(USE_GPU)
#include "_cuda.h"
#endif


#define NUM_CONV_LAYER_PARAMS 2
#define CONV_LAYER_WEIGHTS_PARAM 0
#define CONV_LAYER_BIAS_PARAM 1


#define conv_output_size(input_size, kernel_size, stride, dilation, padding) \
    (((input_size) + 2 * (padding) - (dilation) * ((kernel_size) - 1) - 1) / (stride) + 1)
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define div_ceil(a, b) (((a) + (b) - 1) / (b))


const convolutional_layer_create_info_t conv_default_config = {
    .output_channels = 0,
    .filter_height = 0,
    .filter_width = 0,
    .stride_y = 1,
    .stride_x = 1,
    .padding_y = 0,
    .padding_x = 0,
    .weight_init = winit_xavier,
    .bias_init = winit_zeros,
};


typedef struct convolutional_layer_t {
    tensor_t weights;
    tensor_t bias;
    tensor_t d_weights;
    tensor_t d_bias;

    size_t stride_y;
    size_t stride_x;
    size_t padding_y;
    size_t padding_x;

    device_t device;

    layer_param_ref_t param_refs[NUM_CONV_LAYER_PARAMS];
} convolutional_layer_t;


static uint32_t conv_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape,
    device_t device
)
{
    convolutional_layer_t* conv_layer = (convolutional_layer_t*)context;
    convolutional_layer_create_info_t* conv_create_info =
        (convolutional_layer_create_info_t*)create_info;

    conv_layer->device = device;

    tensor_shape_t weights_shape = {
        .dims[CONV_WEIGHT_OUTPUT_CHANNEL_DIM] = output_shape->dims[TENSOR_CHANNEL_DIM],
        .dims[CONV_WEIGHT_INPUT_CHANNEL_DIM] = input_shape->dims[TENSOR_CHANNEL_DIM],
        .dims[CONV_WEIGHT_HEIGHT_DIM] = conv_create_info->filter_height,
        .dims[CONV_WEIGHT_WIDTH_DIM] = conv_create_info->filter_width
    };
    tensor_allocate_device(&conv_layer->weights, &weights_shape, device);
    tensor_allocate_device(&conv_layer->d_weights, &weights_shape, device);

    tensor_shape_t bias_shape = {
        .dims[0] = conv_create_info->output_channels,
        .dims[1] = 0,
        .dims[2] = 0,
        .dims[3] = 0,
    };
    tensor_allocate_device(&conv_layer->bias, &bias_shape, device);
    tensor_allocate_device(&conv_layer->d_bias, &bias_shape, device);

    conv_layer->stride_y = conv_create_info->stride_y ? conv_create_info->stride_y : 1;
    conv_layer->stride_x = conv_create_info->stride_x ? conv_create_info->stride_x : 1;
    conv_layer->padding_y = conv_create_info->padding_y;
    conv_layer->padding_x = conv_create_info->padding_x;

    /* need to register the params for the optimizer */
    conv_layer->param_refs[CONV_LAYER_WEIGHTS_PARAM].param = &conv_layer->weights;
    conv_layer->param_refs[CONV_LAYER_WEIGHTS_PARAM].gradient = &conv_layer->d_weights;
    conv_layer->param_refs[CONV_LAYER_BIAS_PARAM].param = &conv_layer->bias;
    conv_layer->param_refs[CONV_LAYER_BIAS_PARAM].gradient = &conv_layer->d_bias;

    /* Initialise weights and bias */
    if (device == device_gpu) {
        tensor_t tmp_weights;
        tensor_t tmp_bias;
        tensor_allocate_device(&tmp_weights, &weights_shape, device_cpu);
        tensor_allocate_device(&tmp_bias, &bias_shape, device_cpu);
        conv_create_info->weight_init(&tmp_weights);
        conv_create_info->bias_init(&tmp_bias);
        tensor_copy(&conv_layer->weights, &tmp_weights);
        tensor_copy(&conv_layer->bias, &tmp_bias);
        tensor_destory(&tmp_weights);
        tensor_destory(&tmp_bias);
    } else {
        conv_create_info->weight_init(&conv_layer->weights);
        conv_create_info->bias_init(&conv_layer->bias);
    }

    return 0;
}


static uint32_t conv_layer_get_params(
    layer_context_t* context,
    layer_param_ref_list_t* out_layer_params
)
{
    convolutional_layer_t* conv_layer = (convolutional_layer_t*)context;

    out_layer_params->param_refs = conv_layer->param_refs;
    out_layer_params->num_params = NUM_CONV_LAYER_PARAMS;
    return 0;
}


static uint32_t conv_layer_deinit(layer_context_t* context)
{
    convolutional_layer_t* conv_layer = (convolutional_layer_t*)context;

    tensor_destory(&conv_layer->weights);
    tensor_destory(&conv_layer->d_weights);
    tensor_destory(&conv_layer->bias);
    tensor_destory(&conv_layer->d_bias);
}


static void conv2d(const float* input, const float* kernel, float* output, int32_t input_height,
    int32_t input_width, int32_t kernel_height, int32_t kernel_width, int32_t stride_y,
    int32_t stride_x, int32_t padding_y, int32_t padding_x, int32_t dilation_y, int32_t dilation_x,
    int32_t skip_output_y, int32_t skip_output_x, int32_t flip_kernel, device_t device)
{
    if (device == device_cpu) {
        conv2d_cpu(input, kernel, output, input_height, input_width, kernel_height, kernel_width,
            stride_y, stride_x, padding_y, padding_x, dilation_y, dilation_x, skip_output_y,
            skip_output_x, flip_kernel);
    } else if (device == device_gpu) {
#if defined(USE_GPU)
        conv2d_gpu(input, kernel, output, input_height, input_width, kernel_height, kernel_width,
            stride_y, stride_x, padding_y, padding_x, dilation_y, dilation_x, skip_output_y,
            skip_output_x, flip_kernel);
#else
        LOG_ERROR("Invalid device\n");
#endif
    }
}


static uint32_t conv_layer_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t* out_output
)
{
    convolutional_layer_t* conv_layer = (convolutional_layer_t*)context;

    tensor_set_zero(out_output);

    if (conv_layer->device == device_cpu) {
        convolution_forward_cpu(input, &conv_layer->weights, &conv_layer->bias, out_output,
            conv_layer->stride_y, conv_layer->stride_x, conv_layer->padding_y,
            conv_layer->padding_x, 1, 1, 0, 0, false);
    } else {
#if defined(USE_GPU)
        convolution_forward_gpu(input, &conv_layer->weights, &conv_layer->bias, out_output,
            conv_layer->stride_y, conv_layer->stride_x, conv_layer->padding_y,
            conv_layer->padding_x, 1, 1, 0, 0, false);
#endif
    }
}

static uint32_t conv_layer_backward(
    layer_context_t* context,
    const tensor_t* input,
    const tensor_t* output,
    const tensor_t* prev_gradient,
    tensor_t* out_gradient
)
{
    convolutional_layer_t* conv_layer = (convolutional_layer_t*)context;


    const tensor_shape_t* input_shape = tensor_get_shape(input);
    const tensor_shape_t* output_shape = tensor_get_shape(output);
    const tensor_shape_t* weights_shape = tensor_get_shape(&conv_layer->weights);

    const float* x = tensor_get_data_const(input);
    const float* y = tensor_get_data_const(output);
    const float* dy = tensor_get_data_const(prev_gradient);
    float* w = tensor_get_data(&conv_layer->weights);
    float* b = tensor_get_data(&conv_layer->bias);
    float* dw = tensor_get_data(&conv_layer->d_weights);
    float* db = tensor_get_data(&conv_layer->d_bias);
    float* dx = tensor_get_data(out_gradient);
    

    const size_t batch_size = input_shape->dims[TENSOR_BATCH_DIM];
    const size_t input_channels = input_shape->dims[TENSOR_CHANNEL_DIM];
    const size_t input_height = input_shape->dims[TENSOR_HEIGHT_DIM];
    const size_t input_width = input_shape->dims[TENSOR_WIDTH_DIM];
    const size_t output_channels = output_shape->dims[TENSOR_CHANNEL_DIM];
    const size_t output_height = output_shape->dims[TENSOR_HEIGHT_DIM];
    const size_t output_width = output_shape->dims[TENSOR_WIDTH_DIM];
    const size_t filter_height = weights_shape->dims[CONV_WEIGHT_WIDTH_DIM];
    const size_t filter_width = weights_shape->dims[CONV_WEIGHT_WIDTH_DIM];

    const size_t input_size = input_width * input_height;
    const size_t output_size = output_width * output_height;
    const size_t filter_size = filter_height * filter_width * input_channels;


    // Calculate gradients with respect to the input and store in dx
    tensor_set_zero(out_gradient);
    for (size_t n = 0; n < batch_size; n++) {
        for (size_t i = 0; i < input_channels; i++) {
            float* _dx = dx + n * input_size * input_channels + i * input_size;
            for (size_t j = 0; j < output_channels; j++) {
                const float* _dy = dy + n * output_size * output_channels + j * output_size;
                const float* _w = w + j + filter_size + i * filter_height * filter_width;
                /* dx = conv2d(w, flip(dy),
                            dilation: (stride_y,stride_x),
                            padding: (output_height-1,output_width-1)) */
                conv2d(_w, _dy, _dx, filter_height, filter_width, output_height, output_width,
                    1, 1, output_height - 1, output_width - 1, conv_layer->stride_y,
                    conv_layer->stride_x, conv_layer->padding_y, conv_layer->padding_x, true,
                    conv_layer->device);
            }
        }
    }

    // Compute weight gradients
    tensor_set_zero(&conv_layer->d_weights);
    for (size_t n = 0; n < batch_size; n++) {
        for (size_t i = 0; i < input_channels; i++) {
            const float* _x = x + n * input_size * input_channels + i * input_size;
            for (size_t j = 0; j < output_channels; j++) {
                const float* _dy = dy + n * output_size * output_channels + j * output_size;
                float* _dw = dw + j * filter_size + i * filter_height * filter_width;
                /* dw = conv2d(x, dy, dilation: (stride_y, stride_x)) */
                conv2d(_x, _dy, _dw, input_height, input_width, output_height, output_width, 1, 1,
                    conv_layer->padding_y, conv_layer->padding_x, conv_layer->stride_y,
                    conv_layer->stride_x, 0, 0, false, conv_layer->device);
            }
        }
    }
    tensor_scale(&conv_layer->d_weights, 1.0 / batch_size);

    // Adjust output channel bias
    tensor_set_zero(&conv_layer->d_bias);
    for (size_t n = 0; n < batch_size; n++) {
        const float* _dy = dy + n * output_channels * output_size;
        for (size_t i = 0; i < output_channels; i++) {
            const tensor_t out_channel_grad = {
                .shape = make_tensor_shape(1, output_size),
                .data = _dy + i * output_size,
                .device = conv_layer->device
            };
            tensor_t db_tensor = {
                .shape = make_tensor_shape(1, 1),
                .data = db + i,
                .device = conv_layer->device
            };
            tensor_sum(&db_tensor, &out_channel_grad);
        }
    }
    tensor_scale(&conv_layer->d_bias, 1.0 / batch_size);
}


static uint32_t conv_layer_calc_output_shape(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
)
{
    convolutional_layer_create_info_t* conv_create_info =
        (convolutional_layer_create_info_t*)create_info;

    out_output_shape->dims[TENSOR_BATCH_DIM] = input_shape->dims[TENSOR_BATCH_DIM];
    out_output_shape->dims[TENSOR_CHANNEL_DIM] = conv_create_info->output_channels;
    out_output_shape->dims[TENSOR_HEIGHT_DIM] = conv_output_size(
        input_shape->dims[TENSOR_HEIGHT_DIM], conv_create_info->filter_height,
        conv_create_info->stride_y, 1, conv_create_info->padding_y);
    out_output_shape->dims[TENSOR_WIDTH_DIM] = conv_output_size(input_shape->dims[TENSOR_WIDTH_DIM],
        conv_create_info->filter_width, conv_create_info->stride_x, 1, conv_create_info->padding_x);
    return 0;
}


const layer_impl_t convolutional_layer_impl = {
    .init_func = conv_layer_init,
    .get_param_func = conv_layer_get_params,
    .deinit_func = conv_layer_deinit,
    .forward_func = conv_layer_forward,
    .backward_func = conv_layer_backward,
    .calc_output_size = conv_layer_calc_output_shape,
    .layer_context_size = sizeof(convolutional_layer_t)
};
