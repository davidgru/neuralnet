#include <stdlib.h>
#include <string.h>

#include "log.h"

#include "util/ai_math.h"

#include "ai_convolutional_layer.h"
#include "ai_layer.h"


#define CONV_WEIGHT_OUTPUT_CHANNEL_DIM  0
#define CONV_WEIGHT_INPUT_CHANNEL_DIM   1
#define CONV_WEIGHT_HEIGHT_DIM          2
#define CONV_WEIGHT_WIDTH_DIM           3


typedef struct convolutional_layer_t {
    tensor_t weights;
    tensor_t bias;
    tensor_t d_weights;
    tensor_t d_bias;
    float learning_rate;
} convolutional_layer_t;


static uint32_t conv_layer_init(void* private_data, const AI_LayerCreateInfo* create_info,
    const tensor_shape_t* input_shape, const tensor_shape_t* output_shape);
static uint32_t conv_layer_deinit(void* private_data);
static uint32_t conv_layer_forward(void* private_data, const tensor_t* input,
    tensor_t* out_output);
static uint32_t conv_layer_backward(void* private_data, const tensor_t* input, const tensor_t* output,
    const tensor_t* prev_gradient, tensor_t* out_gradient);
uint32_t conv_layer_calc_output_shape(tensor_shape_t* out_output_shape, const void* create_info,
    const tensor_shape_t* input_shape);


const layer_info_t convolutional_layer_info = {
    .init_func = conv_layer_init,
    .deinit_func = conv_layer_deinit,
    .forward_func = conv_layer_forward,
    .backward_func = conv_layer_backward,
    .calc_output_size = conv_layer_calc_output_shape,
    .info_func = NULL,
    .layer_private_size = sizeof(convolutional_layer_t)
};


static uint32_t conv_layer_init(
    void* private_data,
    const AI_LayerCreateInfo* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
)
{
    convolutional_layer_t* conv_layer = (convolutional_layer_t*)private_data;
    AI_ConvolutionalLayerCreateInfo* conv_create_info =
        (AI_ConvolutionalLayerCreateInfo*)create_info;


    tensor_shape_t weights_shape = {
        .dims[CONV_WEIGHT_OUTPUT_CHANNEL_DIM] = output_shape->dims[TENSOR_CHANNEL_DIM],
        .dims[CONV_WEIGHT_INPUT_CHANNEL_DIM] = input_shape->dims[TENSOR_CHANNEL_DIM],
        .dims[CONV_WEIGHT_HEIGHT_DIM] = conv_create_info->filter_width,
        .dims[CONV_WEIGHT_WIDTH_DIM] = conv_create_info->filter_width
    };
    tensor_allocate(&conv_layer->weights, &weights_shape);
    tensor_allocate(&conv_layer->d_weights, &weights_shape);

    tensor_shape_t bias_shape = {
        .dims[0] = 0,
        .dims[1] = 0,
        .dims[2] = 0,
        .dims[3] = conv_create_info->output_channels
    };
    tensor_allocate(&conv_layer->bias, &bias_shape);
    tensor_allocate(&conv_layer->d_bias, &bias_shape);

    conv_layer->learning_rate = conv_create_info->learning_rate;


    /* initialize weights and bias */

    float* weights_data = tensor_get_data(&conv_layer->weights);
    const size_t weights_size = tensor_size_from_shape(&weights_shape);
    for (size_t i = 0; i < weights_size; i++) {
        weights_data[i] = conv_create_info->weight_init(
            input_shape->dims[TENSOR_WIDTH_DIM], 
            input_shape->dims[TENSOR_HEIGHT_DIM], 
            input_shape->dims[TENSOR_CHANNEL_DIM]
        );
    }

    float* bias_data = tensor_get_data(&conv_layer->bias);
    const size_t bias_size = tensor_size_from_shape(&bias_shape);
    for (size_t i = 0; i < bias_size; i++) {
        bias_data[i] = conv_create_info->bias_init(
            input_shape->dims[TENSOR_WIDTH_DIM], 
            input_shape->dims[TENSOR_HEIGHT_DIM], 
            input_shape->dims[TENSOR_CHANNEL_DIM]
        );
    }

    return 0;
}


static uint32_t conv_layer_deinit(void* private_data)
{
    convolutional_layer_t* conv_layer = (convolutional_layer_t*)private_data;

    tensor_destory(&conv_layer->weights);
    tensor_destory(&conv_layer->d_weights);
    tensor_destory(&conv_layer->bias);
    tensor_destory(&conv_layer->d_bias);
}


static uint32_t conv_layer_forward(
    void* private_data,
    const tensor_t* input,
    tensor_t* out_output
)
{
    LOG_TRACE("conv layer fwd pass\n");

    convolutional_layer_t* conv_layer = (convolutional_layer_t*)private_data;


    const tensor_shape_t* input_shape = tensor_get_shape(input);
    const tensor_shape_t* output_shape = tensor_get_shape(out_output);
    const tensor_shape_t* weights_shape = tensor_get_shape(&conv_layer->weights);


    const float* x = tensor_get_data_const(input);
    const float* w = tensor_get_data_const(&conv_layer->weights);
    const float* b = tensor_get_data_const(&conv_layer->bias);
    float* y = tensor_get_data(out_output);

    const size_t batch_size = input_shape->dims[TENSOR_BATCH_DIM];
    const size_t input_channels = input_shape->dims[TENSOR_CHANNEL_DIM];
    const size_t input_height = input_shape->dims[TENSOR_HEIGHT_DIM];
    const size_t input_width = input_shape->dims[TENSOR_WIDTH_DIM];
    const size_t output_channels = output_shape->dims[TENSOR_CHANNEL_DIM];
    const size_t output_height = output_shape->dims[TENSOR_HEIGHT_DIM];
    const size_t output_width = output_shape->dims[TENSOR_WIDTH_DIM];
    const size_t filter_width = weights_shape->dims[CONV_WEIGHT_WIDTH_DIM];

    const size_t output_size = output_width * output_height;
    const size_t filter_size = filter_width * filter_width * input_channels;


    memset(y, 0, output_size * output_channels * batch_size * sizeof(float));
    for (size_t n = 0; n < batch_size; n++) {
        for (size_t i = 0; i < output_channels; i++) {
            float* _y = y + n * output_size * output_channels + i * output_size;
            for (size_t j = 0; j < input_channels; j++) {
                const float* _x = x + n * input_width * input_height * input_channels + j * input_width * input_height;
                const float* _w = w + i * filter_size + j * filter_width * filter_width;
                // Do a convolution with the one input channel and one filter channel to produce part of one output feature map.
                AI_MatrixConvolution(_x, _w, _y, input_width, input_height, filter_width, filter_width, 1, 1);
            }
            // Add the bias to every element of the feature map
            AI_VectorAddScalar(y, b[i], output_size);
        }
    }
}


static uint32_t conv_layer_backward(
    void* private_data,
    const tensor_t* input,
    const tensor_t* output,
    const tensor_t* prev_gradient,
    tensor_t* out_gradient
)
{
    convolutional_layer_t* conv_layer = (convolutional_layer_t*)private_data;


    const tensor_shape_t* input_shape = tensor_get_shape(input);
    const tensor_shape_t* output_shape = tensor_get_shape(output);
    const tensor_shape_t* weights_shape = tensor_get_shape(&conv_layer->weights);

    const float* x = tensor_get_data_const(input);
    const float* y = tensor_get_data_const(output);
    const float* dy = tensor_get_data_const(prev_gradient);
    float* w = tensor_get_data(&conv_layer->weights);
    float* b = tensor_get_data(&conv_layer->bias);
    float* dw = tensor_get_data(&conv_layer->d_weights);
    float* dx = tensor_get_data(out_gradient);
    

    const size_t batch_size = input_shape->dims[TENSOR_BATCH_DIM];
    const size_t input_channels = input_shape->dims[TENSOR_CHANNEL_DIM];
    const size_t input_height = input_shape->dims[TENSOR_HEIGHT_DIM];
    const size_t input_width = input_shape->dims[TENSOR_WIDTH_DIM];
    const size_t output_channels = output_shape->dims[TENSOR_CHANNEL_DIM];
    const size_t output_height = output_shape->dims[TENSOR_HEIGHT_DIM];
    const size_t output_width = output_shape->dims[TENSOR_WIDTH_DIM];
    const size_t filter_width = weights_shape->dims[CONV_WEIGHT_WIDTH_DIM];

    const size_t input_size = input_width * input_height;
    const size_t output_size = output_width * output_height;
    const size_t filter_size = filter_width * filter_width * input_channels;


    // Calculate gradients with respect to the input and store in dx
    memset(dx, 0, input_size * input_channels * batch_size * sizeof(float));
    for (size_t n = 0; n < batch_size; n++) {
        for (size_t i = 0; i < input_channels; i++) {
            float* _dx = dx + n * input_size * input_channels + i * input_size;
            for (size_t j = 0; j < output_channels; j++) {
                const float* _dy = dy + n * output_size * output_channels + j * output_size;
                const float* _w = w + j + filter_size + i * filter_width * filter_width;
                AI_MatrixConvolutionPaddedRotateFilter(_dy, _w, _dx, output_width, output_height,
                    filter_width, filter_width, 1, 1, filter_width - 1, filter_width - 1, 0, 0);
            }
        }
    }

    // Adjust filter weights
    memset(dw, 0, filter_size * output_channels * sizeof(float));
    for (size_t n = 0; n < batch_size; n++) {
        for (size_t i = 0; i < input_channels; i++) {
            const float* _x = x + n * input_size * input_channels + i * input_size;
            for (size_t j = 0; j < output_channels; j++) {
                const float* _dy = dy + n * output_size * output_channels + j * output_size;
                float* _dw = dw + j * filter_size + i * filter_width * filter_width;
                AI_MatrixConvolutionPadded(_x, _dy, _dw, input_width, input_height, output_width,
                    output_height, 1, 1, 0, 0, 0, 0);
            }
        }
    }
    AI_VectorScale(dw, conv_layer->learning_rate * (1.0 / batch_size), filter_size * output_channels);
    AI_VectorSub(w, dw, filter_size * output_channels);

    // Adjust output channel bias
    for (size_t n = 0; n < batch_size; n++) {
        const float* _dy = dy + n * output_channels * output_size;
        for (size_t i = 0; i < output_channels; i++) {
            float _db = AI_Sum(_dy + i * output_size, output_size);
            b[i] -= conv_layer->learning_rate * (1.0 / batch_size) * _db;
        }
    }
}


uint32_t conv_layer_calc_output_shape(
    tensor_shape_t* out_output_shape,
    const void* create_info,
    const tensor_shape_t* input_shape
)
{
    AI_ConvolutionalLayerCreateInfo* conv_create_info =
        (AI_ConvolutionalLayerCreateInfo*)create_info;

    out_output_shape->dims[TENSOR_BATCH_DIM] = input_shape->dims[TENSOR_BATCH_DIM];
    out_output_shape->dims[TENSOR_CHANNEL_DIM] = conv_create_info->output_channels;
    out_output_shape->dims[TENSOR_HEIGHT_DIM] = input_shape->dims[TENSOR_HEIGHT_DIM]
        - conv_create_info->filter_width + 1;
    out_output_shape->dims[TENSOR_WIDTH_DIM] = input_shape->dims[TENSOR_WIDTH_DIM]
        - conv_create_info->filter_width + 1;
    
    return 0;
}