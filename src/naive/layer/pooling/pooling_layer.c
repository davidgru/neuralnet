#include <malloc.h>
#include <string.h>

#include "tensor/tensor_impl.h"
#include "layer/pooling_layer.h"
#include "pooling_layer_internal.h"

#define max(a, b) ((a > b) ? a : b)
#define min(a, b) ((a < b) ? a : b)


typedef struct pooling_layer_t {
    size_t kernel_width;
    device_t device;
    pooling_kind_t kind;
} pooling_layer_t;


static uint32_t pooling_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape,
    device_t device
)
{
    pooling_layer_t* pooling_layer = (pooling_layer_t*)context;
    pooling_layer_create_info_t* pooling_create_info = (pooling_layer_create_info_t*)create_info;

    pooling_layer->device = device;
    pooling_layer->kernel_width = pooling_create_info->kernel_width;
    pooling_layer->kind = pooling_create_info->pooling_operation;

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
    
    tensor_set_zero(out_output);
    if (pooling_layer->device == device_cpu) {
        pooling_forward_cpu(input, out_output, pooling_layer->kernel_width, pooling_layer->kind);
    } else {
#if defined(USE_GPU)
        pooling_forward_gpu(input, out_output, pooling_layer->kernel_width, pooling_layer->kind);
#endif
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

    tensor_set_zero(out_gradient);
    if (pooling_layer->device == device_cpu) {
        pooling_backward_cpu(input, prev_gradient, out_gradient, pooling_layer->kernel_width, pooling_layer->kind);
    } else {
#if defined(USE_GPU)
        pooling_backward_gpu(input, prev_gradient, out_gradient, pooling_layer->kernel_width, pooling_layer->kind);
#endif
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



const layer_impl_t pooling_layer_impl = {
    .init_func = pooling_layer_init,
    .get_param_func = NULL, /* no params */
    .deinit_func = NULL,
    .forward_func = pooling_layer_forward,
    .backward_func = pooling_layer_backward,
    .calc_output_size = pooling_layer_calc_output_shape,
    .layer_context_size = sizeof(pooling_layer_t),
};
