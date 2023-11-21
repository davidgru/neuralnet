#include <stdlib.h>
#include <string.h>

#include "log.h"

#include "util/ai_math.h"

#include "ai_convolutional_layer.h"


#define CONV_WEIGHT_OUTPUT_CHANNEL_DIM  0
#define CONV_WEIGHT_INPUT_CHANNEL_DIM   1
#define CONV_WEIGHT_HEIGHT_DIM          2
#define CONV_WEIGHT_WIDTH_DIM           3


typedef struct convolutional_layer_t {
    AI_Layer hdr;
    tensor_t weights;
    tensor_t bias;
    tensor_t d_weights;
    tensor_t d_bias;
    float learning_rate;
} convolutional_layer_t;




static void conv_layer_forward(AI_Layer* layer);
static void conv_layer_backward(AI_Layer* layer);
static void conv_layer_deinit(AI_Layer* layer);

static tensor_shape_t calculate_output_shape(
    const tensor_shape_t* input_shape,
    const AI_ConvolutionalLayerCreateInfo* create_info
);

uint32_t convolutional_layer_init(AI_Layer** layer, void* create_info, AI_Layer* prev_layer)
{
    AI_ConvolutionalLayerCreateInfo* conv_create_info =
        (AI_ConvolutionalLayerCreateInfo*)create_info;

    *layer = (AI_Layer*)malloc(sizeof(convolutional_layer_t));
    if (*layer == NULL) {
        return 1;
    }

    convolutional_layer_t* conv_layer = (convolutional_layer_t*)*layer;


    /* fill header information */
    
    conv_layer->hdr.input_shape = prev_layer->output_shape;
    conv_layer->hdr.output_shape = calculate_output_shape(&conv_layer->hdr.input_shape,
        conv_create_info);

    /* allocate owned memory */
    tensor_allocate(&conv_layer->hdr.gradient, &conv_layer->hdr.input_shape);
    tensor_allocate(&conv_layer->hdr.output, &conv_layer->hdr.output_shape);
    
    tensor_shape_t weights_shape = {
        .dims[CONV_WEIGHT_OUTPUT_CHANNEL_DIM] = conv_create_info->output_channels,
        .dims[CONV_WEIGHT_INPUT_CHANNEL_DIM] = conv_layer->hdr.input_shape.dims[TENSOR_CHANNEL_DIM],
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

    /* virtual functions */
    conv_layer->hdr.forward = conv_layer_forward;
    conv_layer->hdr.backward = conv_layer_backward;
    conv_layer->hdr.info = NULL;
    conv_layer->hdr.deinit = conv_layer_deinit;

    conv_layer->learning_rate = conv_create_info->learning_rate;


    /* initialize weights and bias */

    float* weights_data = tensor_get_data(&conv_layer->weights);
    const size_t weights_size = tensor_size_from_shape(&weights_shape);
    for (size_t i = 0; i < weights_size; i++) {
        weights_data[i] = conv_create_info->weight_init(
            conv_layer->hdr.input_shape.dims[TENSOR_WIDTH_DIM], 
            conv_layer->hdr.input_shape.dims[TENSOR_HEIGHT_DIM], 
            conv_layer->hdr.input_shape.dims[TENSOR_CHANNEL_DIM]
        );
    }

    float* bias_data = tensor_get_data(&conv_layer->bias);
    const size_t bias_size = tensor_size_from_shape(&bias_shape);
    for (size_t i = 0; i < bias_size; i++) {
        bias_data[i] = conv_create_info->bias_init(
            conv_layer->hdr.input_shape.dims[TENSOR_WIDTH_DIM], 
            conv_layer->hdr.input_shape.dims[TENSOR_HEIGHT_DIM], 
            conv_layer->hdr.input_shape.dims[TENSOR_CHANNEL_DIM]
        );
    }

    return 0;
}


static void conv_layer_forward(AI_Layer* layer)
{
    LOG_TRACE("conv layer fwd pass\n");

    convolutional_layer_t* conv_layer = (convolutional_layer_t*)layer;


    const tensor_shape_t* input_shape = tensor_get_shape(conv_layer->hdr.input);
    const tensor_shape_t* output_shape = tensor_get_shape(&conv_layer->hdr.output);
    const tensor_shape_t* weights_shape = tensor_get_shape(&conv_layer->weights);


    float* x = tensor_get_data(conv_layer->hdr.input);
    float* y = tensor_get_data(&conv_layer->hdr.output);
    float* w = tensor_get_data(&conv_layer->weights);
    float* b = tensor_get_data(&conv_layer->bias);


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
    convolutional_layer_t* conv_layer = (convolutional_layer_t*)layer;


    const tensor_shape_t* input_shape = tensor_get_shape(conv_layer->hdr.input);
    const tensor_shape_t* output_shape = tensor_get_shape(&conv_layer->hdr.output);
    const tensor_shape_t* weights_shape = tensor_get_shape(&conv_layer->weights);

    float* x = tensor_get_data(conv_layer->hdr.input);
    float* y = tensor_get_data(&conv_layer->hdr.output);
    float* w = tensor_get_data(&conv_layer->weights);
    float* b = tensor_get_data(&conv_layer->bias);
    float* dx = tensor_get_data(&conv_layer->hdr.gradient);
    float* dy = tensor_get_data(conv_layer->hdr.prev_gradient);
    float* dw = tensor_get_data(&conv_layer->d_weights);
    

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
                float* _dy = dy + n * output_size * output_channels + j * output_size;
                float* _w = w + j + filter_size + i * filter_width * filter_width;
                AI_MatrixConvolutionPaddedRotateFilter(_dy, _w, _dx, output_width, output_height,
                    filter_width, filter_width, 1, 1, filter_width - 1, filter_width - 1, 0, 0);
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
                AI_MatrixConvolutionPadded(_x, _dy, _dw, input_width, input_height, output_width,
                    output_height, 1, 1, 0, 0, 0, 0);
            }
        }
    }
    AI_VectorScale(dw, conv_layer->learning_rate, filter_size * output_channels);
    AI_VectorSub(w, dw, filter_size * output_channels);

    // Adjust output channel bias
    for (size_t i = 0; i < output_channels; i++) {
        float _db = AI_Sum(dy + i * output_size, output_size);
        b[i] -= conv_layer->learning_rate * _db;
    }
}


static void conv_layer_deinit(AI_Layer* layer)
{
    convolutional_layer_t* conv_layer = (convolutional_layer_t*)layer;

    if (conv_layer != NULL) {
        tensor_destory(&conv_layer->hdr.output);
        tensor_destory(&conv_layer->hdr.gradient);
        tensor_destory(&conv_layer->weights);
        tensor_destory(&conv_layer->d_weights);
        tensor_destory(&conv_layer->bias);
        tensor_destory(&conv_layer->d_bias);
        free(conv_layer);
    }
}


static tensor_shape_t calculate_output_shape(
    const tensor_shape_t* input_shape,
    const AI_ConvolutionalLayerCreateInfo* create_info
)
{
    tensor_shape_t output_shape;
    output_shape.dims[TENSOR_BATCH_DIM] = input_shape->dims[TENSOR_BATCH_DIM];
    output_shape.dims[TENSOR_CHANNEL_DIM] = create_info->output_channels;
    output_shape.dims[TENSOR_HEIGHT_DIM] = input_shape->dims[TENSOR_HEIGHT_DIM]
        - create_info->filter_width + 1;
    output_shape.dims[TENSOR_WIDTH_DIM] = input_shape->dims[TENSOR_WIDTH_DIM]
        - create_info->filter_width + 1;
    return output_shape;
}
