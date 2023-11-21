
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

uint32_t pooling_layer_init(AI_Layer** layer, void* create_info, AI_Layer* prev_layer)
{
    AI_PoolingLayerCreateInfo* pooling_create_info = (AI_PoolingLayerCreateInfo*)create_info;

    *layer = (AI_Layer*)malloc(sizeof(pooling_layer_t));
    if (*layer == NULL) {
        return 1;
    }

    pooling_layer_t* pooling_layer = (pooling_layer_t*)*layer;

    /* fill header information */
    pooling_layer->hdr.input_shape = prev_layer->output_shape;
    pooling_layer->hdr.output_shape.dims[TENSOR_BATCH_DIM]
        = pooling_layer->hdr.input_shape.dims[TENSOR_BATCH_DIM];
    pooling_layer->hdr.output_shape.dims[TENSOR_CHANNEL_DIM]
        = pooling_layer->hdr.input_shape.dims[TENSOR_CHANNEL_DIM];
    pooling_layer->hdr.output_shape.dims[TENSOR_HEIGHT_DIM]
        = pooling_layer->hdr.input_shape.dims[TENSOR_HEIGHT_DIM]
        / pooling_create_info->kernel_width;
    pooling_layer->hdr.output_shape.dims[TENSOR_WIDTH_DIM]
        = pooling_layer->hdr.input_shape.dims[TENSOR_WIDTH_DIM]
        / pooling_create_info->kernel_width;
    
    /* allocate owned memory */
    tensor_allocate(&pooling_layer->hdr.output, &pooling_layer->hdr.output_shape);
    tensor_allocate(&pooling_layer->hdr.gradient, &pooling_layer->hdr.input_shape);

    /* virtual functions */
    pooling_layer->hdr.forward = pooling_layer_forward;
    pooling_layer->hdr.backward = pooling_layer_backward;
    pooling_layer->hdr.info = NULL;
    pooling_layer->hdr.deinit = pooling_layer_deinit;

    pooling_layer->kernel_width = pooling_create_info->kernel_width;

    switch(pooling_create_info->pooling_operation) {
        case AI_POOLING_AVERAGE:
            pooling_layer->pooling_operation_func = pooling_operation_average;
            pooling_layer->pooling_operation_backward = pooling_operation_average_backward;
            break;
        case AI_POOLING_MAX:
            pooling_layer->pooling_operation_func = pooling_operation_max;
            pooling_layer->pooling_operation_backward = pooling_operation_max_backward;
            break;
        case AI_POOLING_MIN:
            pooling_layer->pooling_operation_func = pooling_operation_min;
            pooling_layer->pooling_operation_backward = pooling_operation_min_backward;
            break;
    }
}



static void pooling_layer_forward(AI_Layer* layer)
{
    pooling_layer_t* pooling_layer = (pooling_layer_t*)layer;


    const tensor_shape_t* input_shape = tensor_get_shape(pooling_layer->hdr.input);
    const tensor_shape_t* output_shape = tensor_get_shape(&pooling_layer->hdr.output);


    float* input_data = tensor_get_data(pooling_layer->hdr.input);
    float* output_data = tensor_get_data(&pooling_layer->hdr.output);
    

    const size_t batch_size = input_shape->dims[TENSOR_BATCH_DIM];
    const size_t channels = input_shape->dims[TENSOR_CHANNEL_DIM];
    const size_t input_height = input_shape->dims[TENSOR_HEIGHT_DIM];
    const size_t input_width = input_shape->dims[TENSOR_WIDTH_DIM];
    const size_t per_batch_input_size = tensor_size_from_shape(input_shape) / batch_size;
    const size_t output_height = output_shape->dims[TENSOR_HEIGHT_DIM];
    const size_t output_width = output_shape->dims[TENSOR_WIDTH_DIM];
    const size_t per_batch_output_size = tensor_size_from_shape(output_shape) / batch_size;
    
    memset(output_data, 0, tensor_size_from_shape(output_shape) * sizeof(float));
    for (size_t n = 0; n < batch_size; n++) {
        for (size_t i = 0; i < channels; i++) {
            float* feature_map_in = input_data + n * per_batch_input_size
                + i * input_width * input_height;
            float* feature_map_out = output_data + n * per_batch_output_size
                + i * output_width * output_height;
            pooling_layer->pooling_operation_func(feature_map_in, feature_map_out, input_width,
                input_height, pooling_layer->kernel_width);
        }
    }
}


static void pooling_layer_backward(AI_Layer* layer)
{
    pooling_layer_t* pooling_layer = (pooling_layer_t*)layer;


    const tensor_shape_t* input_shape = tensor_get_shape(pooling_layer->hdr.input);
    const tensor_shape_t* output_shape = tensor_get_shape(&pooling_layer->hdr.output);


    float* input = tensor_get_data(pooling_layer->hdr.input);
    float* gradient = tensor_get_data(&pooling_layer->hdr.gradient);
    float* prev_gradient = tensor_get_data(pooling_layer->hdr.prev_gradient);


    const size_t batch_size = input_shape->dims[TENSOR_BATCH_DIM];
    const size_t channels = input_shape->dims[TENSOR_CHANNEL_DIM];
    const size_t input_height = input_shape->dims[TENSOR_HEIGHT_DIM];
    const size_t input_width = input_shape->dims[TENSOR_WIDTH_DIM];
    const size_t per_batch_input_size = tensor_size_from_shape(input_shape) / batch_size;
    const size_t output_height = output_shape->dims[TENSOR_HEIGHT_DIM];
    const size_t output_width = output_shape->dims[TENSOR_WIDTH_DIM];
    const size_t per_batch_output_size = tensor_size_from_shape(output_shape) / batch_size;


    memset(gradient, 0, tensor_size_from_shape(input_shape) * sizeof(float));
    for (size_t n = 0; n < batch_size; n++) {
        for (size_t i = 0; i < channels; i++) {
            float* feature_map_in = input + n * per_batch_input_size
                + i * input_width * input_height;
            float* d_feature_map_in = gradient + n * per_batch_input_size
                + i * input_width * input_height; 
            float* d_feature_map_out = prev_gradient + n * per_batch_output_size
                + i * output_width * output_height;
            pooling_layer->pooling_operation_backward(feature_map_in, d_feature_map_out,
                d_feature_map_in, input_width, input_height, pooling_layer->kernel_width);
        }
    }
}


static void pooling_layer_deinit(AI_Layer* layer)
{
    pooling_layer_t* pooling_layer = (pooling_layer_t*)layer;
    if (pooling_layer != NULL) {
        tensor_destory(&pooling_layer->hdr.output);
        tensor_destory(&pooling_layer->hdr.gradient);
    }
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
