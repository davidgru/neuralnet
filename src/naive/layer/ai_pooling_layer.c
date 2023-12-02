#include <malloc.h>
#include <string.h>

#include "ai_layer.h"
#include "ai_pooling_layer.h"


#define max(a, b) ((a > b) ? a : b)
#define min(a, b) ((a < b) ? a : b)


typedef struct pooling_layer_t {
    size_t kernel_width;
    void (*pooling_operation_func)(const float* input, float* output, size_t input_width, size_t input_height, size_t kernel_width);
    void (*pooling_operation_backward)(const float* x, const float* dy, float* dx, size_t input_width, size_t input_height, size_t kernel_width);
} pooling_layer_t;


static uint32_t pooling_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
);

static uint32_t pooling_layer_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t* out_output
);

static uint32_t pooling_layer_backward(
    layer_context_t* context,
    const tensor_t* input,
    const tensor_t* output,
    const tensor_t* prev_gradient,
    tensor_t* out_gradient
);

static uint32_t pooling_layer_calc_output_shape(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
);


const layer_impl_t pooling_layer_impl = {
    .init_func = pooling_layer_init,
    .get_param_func = NULL, /* no params */
    .deinit_func = NULL,
    .forward_func = pooling_layer_forward,
    .backward_func = pooling_layer_backward,
    .calc_output_size = pooling_layer_calc_output_shape,
    .layer_context_size = sizeof(pooling_layer_t),
};


static void pooling_operation_average(const float* x, float* y, size_t input_width, size_t input_height, size_t kernel_width);
static void pooling_operation_max(const float* x, float* y, size_t input_width, size_t input_height, size_t kernel_width);
static void pooling_operation_min(const float* x, float* y, size_t input_width, size_t input_height, size_t kernel_width);

static void pooling_operation_average_backward(const float* x, const float* dy, float* dx, size_t input_width, size_t input_height, size_t kernel_width);
static void pooling_operation_max_backward(const float* x, const float* dy, float* dx, size_t input_width, size_t input_height, size_t kernel_width);
static void pooling_operation_min_backward(const float* x, const float* dy, float* dx, size_t input_width, size_t input_height, size_t kernel_width);




static uint32_t pooling_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
)
{
    pooling_layer_t* pooling_layer = (pooling_layer_t*)context;
    pooling_layer_create_info_t* pooling_create_info = (pooling_layer_create_info_t*)create_info;


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

    return 0;
}


static uint32_t pooling_layer_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t* out_output
)
{
    pooling_layer_t* pooling_layer = (pooling_layer_t*)context;


    const tensor_shape_t* input_shape = tensor_get_shape(input);
    const tensor_shape_t* output_shape = tensor_get_shape(out_output);


    const float* input_data = tensor_get_data_const(input);
    float* output_data = tensor_get_data(out_output);
    

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
            const float* feature_map_in = input_data + n * per_batch_input_size
                + i * input_width * input_height;
            float* feature_map_out = output_data + n * per_batch_output_size
                + i * output_width * output_height;
            pooling_layer->pooling_operation_func(feature_map_in, feature_map_out, input_width,
                input_height, pooling_layer->kernel_width);
        }
    }
}


static uint32_t pooling_layer_backward(
    layer_context_t* context,
    const tensor_t* input,
    const tensor_t* output,
    const tensor_t* prev_gradient,
    tensor_t* out_gradient
)
{
    pooling_layer_t* pooling_layer = (pooling_layer_t*)context;


    const tensor_shape_t* input_shape = tensor_get_shape(input);
    const tensor_shape_t* output_shape = tensor_get_shape(output);


    const float* input_data = tensor_get_data_const(input);
    const float* prev_gradient_data = tensor_get_data_const(prev_gradient);
    float* gradient_data = tensor_get_data(out_gradient);


    const size_t batch_size = input_shape->dims[TENSOR_BATCH_DIM];
    const size_t channels = input_shape->dims[TENSOR_CHANNEL_DIM];
    const size_t input_height = input_shape->dims[TENSOR_HEIGHT_DIM];
    const size_t input_width = input_shape->dims[TENSOR_WIDTH_DIM];
    const size_t per_batch_input_size = tensor_size_from_shape(input_shape) / batch_size;
    const size_t output_height = output_shape->dims[TENSOR_HEIGHT_DIM];
    const size_t output_width = output_shape->dims[TENSOR_WIDTH_DIM];
    const size_t per_batch_output_size = tensor_size_from_shape(output_shape) / batch_size;


    memset(gradient_data, 0, tensor_size_from_shape(input_shape) * sizeof(float));
    for (size_t n = 0; n < batch_size; n++) {
        for (size_t i = 0; i < channels; i++) {
            const float* feature_map_in = input_data + n * per_batch_input_size
                + i * input_width * input_height;
            float* d_feature_map_in = gradient_data + n * per_batch_input_size
                + i * input_width * input_height; 
            const float* d_feature_map_out = prev_gradient_data + n * per_batch_output_size
                + i * output_width * output_height;
            pooling_layer->pooling_operation_backward(feature_map_in, d_feature_map_out,
                d_feature_map_in, input_width, input_height, pooling_layer->kernel_width);
        }
    }
}


static uint32_t pooling_layer_calc_output_shape(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
)
{
    pooling_layer_create_info_t* pooling_create_info = (pooling_layer_create_info_t*)create_info;

    out_output_shape->dims[TENSOR_BATCH_DIM] = input_shape->dims[TENSOR_BATCH_DIM];
    out_output_shape->dims[TENSOR_CHANNEL_DIM] = input_shape->dims[TENSOR_CHANNEL_DIM];
    out_output_shape->dims[TENSOR_HEIGHT_DIM] = input_shape->dims[TENSOR_HEIGHT_DIM]
        / pooling_create_info->kernel_width;
    out_output_shape->dims[TENSOR_WIDTH_DIM] = input_shape->dims[TENSOR_WIDTH_DIM]
        / pooling_create_info->kernel_width;

    return 0;
}


static void pooling_operation_average(const float* x, float* y, size_t input_width, size_t input_height, size_t kernel_width)
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


static void pooling_operation_max(const float* x, float* y, size_t input_width, size_t input_height, size_t kernel_width)
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


static void pooling_operation_min(const float* x, float* y, size_t input_width, size_t input_height, size_t kernel_width)
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


static void pooling_operation_average_backward(const float* x, const float* dy, float* dx, size_t input_width, size_t input_height, size_t kernel_width)
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


static void pooling_operation_max_backward(const float* x, const float* dy, float* dx, size_t input_width, size_t input_height, size_t kernel_width)
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

static void pooling_operation_min_backward(const float* x, const float* dy, float* dx, size_t input_width, size_t input_height, size_t kernel_width)
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
