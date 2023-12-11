#include <stdlib.h>
#include <string.h>

#include "util/ai_math.h"
#include "tensor_impl.h"
#include "log.h"

#include "layer/convolutional_layer.h"


#define NUM_CONV_LAYER_PARAMS 2
#define CONV_LAYER_WEIGHTS_PARAM 0
#define CONV_LAYER_BIAS_PARAM 1


#define CONV_WEIGHT_OUTPUT_CHANNEL_DIM  0
#define CONV_WEIGHT_INPUT_CHANNEL_DIM   1
#define CONV_WEIGHT_HEIGHT_DIM          2
#define CONV_WEIGHT_WIDTH_DIM           3


#define conv_output_size(input_size, kernel_size, stride, dilation, padding) \
    (((input_size) + 2 * (padding) - (dilation) * ((kernel_size) - 1) - 1) / (stride) + 1)
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define div_ceil(a, b) (((a) + (b) - 1) / (b))


typedef struct convolutional_layer_t {
    tensor_t weights;
    tensor_t bias;
    tensor_t d_weights;
    tensor_t d_bias;

    size_t stride_y;
    size_t stride_x;
    size_t padding_y;
    size_t padding_x;

    layer_param_ref_t param_refs[NUM_CONV_LAYER_PARAMS];
} convolutional_layer_t;


static uint32_t conv_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
);

static uint32_t conv_layer_get_params(
    layer_context_t* context,
    layer_param_ref_list_t* out_layer_params
);

static uint32_t conv_layer_deinit(layer_context_t* context);

static uint32_t conv_layer_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t* out_output
);

static uint32_t conv_layer_backward(
    layer_context_t* context,
    const tensor_t* input,
    const tensor_t* output,
    const tensor_t* prev_gradient,
    tensor_t* out_gradient
);

static uint32_t conv_layer_calc_output_shape(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
);



const convolutional_layer_create_info_t conv_default_config = {
    .output_channels = 0,
    .filter_height = 0,
    .filter_width = 0,
    .stride_y = 1,
    .stride_x = 1,
    .padding_y = 0,
    .padding_x = 0,
    .weight_init = conv_weight_init_xavier,
    .bias_init = conv_bias_init_zeros,
};


const layer_impl_t convolutional_layer_impl = {
    .init_func = conv_layer_init,
    .get_param_func = conv_layer_get_params,
    .deinit_func = conv_layer_deinit,
    .forward_func = conv_layer_forward,
    .backward_func = conv_layer_backward,
    .calc_output_size = conv_layer_calc_output_shape,
    .layer_context_size = sizeof(convolutional_layer_t)
};


static void conv2d(const float* input, const float* kernel, float* output, int32_t input_height,
    int32_t input_width, int32_t kernel_height, int32_t kernel_width, int32_t stride_y,
    int32_t stride_x, int32_t padding_y, int32_t padding_x, int32_t dilation_y, int32_t dilation_x,
    int32_t skip_output_y, int32_t skip_output_x);
static void conv2d_flip(const float* input, const float* kernel, float* output,
    int32_t input_height, int32_t input_width, int32_t kernel_height, int32_t kernel_width,
    int32_t stride_y, int32_t stride_x, int32_t padding_y, int32_t padding_x, int32_t dilation_y,
    int32_t dilation_x, int32_t skip_output_y, int32_t skip_output_x);



static uint32_t conv_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
)
{
    convolutional_layer_t* conv_layer = (convolutional_layer_t*)context;
    convolutional_layer_create_info_t* conv_create_info =
        (convolutional_layer_create_info_t*)create_info;


    tensor_shape_t weights_shape = {
        .dims[CONV_WEIGHT_OUTPUT_CHANNEL_DIM] = output_shape->dims[TENSOR_CHANNEL_DIM],
        .dims[CONV_WEIGHT_INPUT_CHANNEL_DIM] = input_shape->dims[TENSOR_CHANNEL_DIM],
        .dims[CONV_WEIGHT_HEIGHT_DIM] = conv_create_info->filter_height,
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

    conv_layer->stride_y = conv_create_info->stride_y;
    conv_layer->stride_x = conv_create_info->stride_x;
    conv_layer->padding_y = conv_create_info->padding_y;
    conv_layer->padding_x = conv_create_info->padding_x;

    /* need to register the params for the optimizer */
    conv_layer->param_refs[CONV_LAYER_WEIGHTS_PARAM].param = &conv_layer->weights;
    conv_layer->param_refs[CONV_LAYER_WEIGHTS_PARAM].gradient = &conv_layer->d_weights;
    conv_layer->param_refs[CONV_LAYER_BIAS_PARAM].param = &conv_layer->bias;
    conv_layer->param_refs[CONV_LAYER_BIAS_PARAM].gradient = &conv_layer->d_bias;


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


static uint32_t conv_layer_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t* out_output
)
{
    convolutional_layer_t* conv_layer = (convolutional_layer_t*)context;


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
    const size_t filter_height = weights_shape->dims[CONV_WEIGHT_WIDTH_DIM];
    const size_t filter_width = weights_shape->dims[CONV_WEIGHT_WIDTH_DIM];

    const size_t output_size = output_width * output_height;
    const size_t filter_size = filter_width * filter_height * input_channels;


    memset(y, 0, output_size * output_channels * batch_size * sizeof(float));
    for (size_t n = 0; n < batch_size; n++) {
        for (size_t i = 0; i < output_channels; i++) {
            float* _y = y + n * output_size * output_channels + i * output_size;
            for (size_t j = 0; j < input_channels; j++) {
                const float* _x = x + n * input_width * input_height * input_channels + j * input_width * input_height;
                const float* _w = w + i * filter_size + j * filter_width * filter_height;
                // Do a convolution with the one input channel and one filter channel to produce part of one output feature map.
                conv2d(_x, _w, _y, input_height, input_height, filter_width, filter_width,
                    conv_layer->stride_y, conv_layer->stride_x, conv_layer->padding_y,
                    conv_layer->padding_x, 1, 1, 0, 0);
            }
            // Add the bias to every element of the feature map
            VectorAddScalar(_y, b[i], output_size);
        }
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
    memset(dx, 0, input_size * input_channels * batch_size * sizeof(float));
    for (size_t n = 0; n < batch_size; n++) {
        for (size_t i = 0; i < input_channels; i++) {
            float* _dx = dx + n * input_size * input_channels + i * input_size;
            for (size_t j = 0; j < output_channels; j++) {
                const float* _dy = dy + n * output_size * output_channels + j * output_size;
                const float* _w = w + j + filter_size + i * filter_height * filter_width;
                /* dx = conv2d(w, flip(dy),
                            dilation: (stride_y,stride_x),
                            padding: (output_height-1,output_width-1)) */
                conv2d_flip(_w, _dy, _dx, filter_height, filter_width, output_height, output_width,
                    1, 1, output_height - 1, output_width - 1, conv_layer->stride_y,
                    conv_layer->stride_x, conv_layer->padding_y, conv_layer->padding_x);
            }
        }
    }

    // Compute weight gradients
    memset(dw, 0, filter_size * output_channels * sizeof(float));
    for (size_t n = 0; n < batch_size; n++) {
        for (size_t i = 0; i < input_channels; i++) {
            const float* _x = x + n * input_size * input_channels + i * input_size;
            for (size_t j = 0; j < output_channels; j++) {
                const float* _dy = dy + n * output_size * output_channels + j * output_size;
                float* _dw = dw + j * filter_size + i * filter_height * filter_width;
                /* dw = conv2d(x, dy, dilation: (stride_y, stride_x)) */
                conv2d(_x, _dy, _dw, input_height, input_width, output_height, output_width, 1, 1,
                    conv_layer->padding_y, conv_layer->padding_x, conv_layer->stride_y,
                    conv_layer->stride_x, 0, 0);
            }
        }
    }
    VectorScale(dw, (1.0 / batch_size), filter_size * output_channels);

    // Adjust output channel bias
    for (size_t n = 0; n < batch_size; n++) {
        const float* _dy = dy + n * output_channels * output_size;
        for (size_t i = 0; i < output_channels; i++) {
            db[i] = (1.0 / batch_size) * Sum(_dy + i * output_size, output_size);
        }
    }
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




static void conv2d(
    const float* input,
    const float* kernel,
    float* output,
    int32_t input_height,
    int32_t input_width,
    int32_t kernel_height,
    int32_t kernel_width,
    int32_t stride_y,
    int32_t stride_x,
    int32_t padding_y,
    int32_t padding_x,
    int32_t dilation_y,
    int32_t dilation_x,
    int32_t skip_output_y,
    int32_t skip_output_x
)
{
    /* in case set to zero as "default" */
    stride_y = stride_y ? stride_y : 1;
    stride_x = stride_x ? stride_x : 1;
    dilation_x = dilation_x ? dilation_x : 1;
    dilation_y = dilation_y ? dilation_y : 1;


    int32_t output_height = conv_output_size(input_height, kernel_height, stride_y, dilation_y,
        padding_y) - 2 * skip_output_y;
    int32_t output_width = conv_output_size(input_width, kernel_width, stride_x, dilation_x,
        padding_x) - 2 * skip_output_x;

    for (int32_t r = 0; r < output_height; r++) {
        for (int32_t c = 0; c < output_width; c++) {
            int32_t data_r = (r + skip_output_y) * stride_y - padding_y;
            int32_t data_c = (c + skip_output_x) * stride_x - padding_x;
        
            /* calculate the bounds of the kernel to skip the elements that are in the padding */
            int32_t kr_start = max(0, div_ceil(-data_r, dilation_y));
            int32_t kr_end = min(kernel_height, div_ceil(input_height - data_r, dilation_y));
            int32_t kc_start = max(0, div_ceil(-data_c, dilation_x));
            int32_t kc_end = min(kernel_width, div_ceil(input_width - data_c, dilation_x));
            
            for (int32_t kr = kr_start; kr < kr_end; kr++) { 
                for (int32_t kc = kc_start; kc < kc_end; kc++) {
                    
                    int32_t data_rk = data_r + kr * dilation_y;
                    int32_t data_ck = data_c + kc * dilation_x;
            
                    output[r * output_width + c] += input[data_rk * input_width + data_ck]
                        * kernel[kr * kernel_width + kc];
                }
            }
        }
    }
}


static void conv2d_flip(
    const float* input,
    const float* kernel,
    float* output,
    int32_t input_height,
    int32_t input_width,
    int32_t kernel_height,
    int32_t kernel_width,
    int32_t stride_y,
    int32_t stride_x,
    int32_t padding_y,
    int32_t padding_x,
    int32_t dilation_y,
    int32_t dilation_x,
    int32_t skip_output_y,
    int32_t skip_output_x
)
{
    /* in case set to zero as "default" */
    stride_y = stride_y ? stride_y : 1;
    stride_x = stride_x ? stride_x : 1;
    dilation_x = dilation_x ? dilation_x : 1;
    dilation_y = dilation_y ? dilation_y : 1;

    int32_t output_height = conv_output_size(input_height, kernel_height, stride_y, dilation_y,
        padding_y) - 2 * skip_output_y;
    int32_t output_width = conv_output_size(input_width, kernel_width, stride_x, dilation_x,
        padding_x) - 2 * skip_output_x;
    
    for (int32_t r = 0; r < output_height; r++) {
        for (int32_t c = 0; c < output_width; c++) {
            int32_t data_r = (r + skip_output_y) * stride_y - padding_y;
            int32_t data_c = (c + skip_output_x) * stride_x - padding_x;
        
            /* calculate the bounds of the kernel to skip the elements that are in the padding */
            int32_t kr_start = max(0, div_ceil(-data_r, dilation_y));
            int32_t kr_end = min(kernel_height, div_ceil(input_height - data_r, dilation_y));
            int32_t kc_start = max(0, div_ceil(-data_c, dilation_x));
            int32_t kc_end = min(kernel_width, div_ceil(input_width - data_c, dilation_x));
            
            for (int32_t kr = kr_start; kr < kr_end; kr++) { 
                for (int32_t kc = kc_start; kc < kc_end; kc++) {
                    
                    int32_t data_rk = data_r + kr * dilation_y;
                    int32_t data_ck = data_c + kc * dilation_x;
            
                    output[r * output_width + c] += input[data_rk * input_width + data_ck]
                        * kernel[(kernel_height - kr - 1) * kernel_width + (kernel_width - kc - 1)];
                }
            }
        }
    }
}
