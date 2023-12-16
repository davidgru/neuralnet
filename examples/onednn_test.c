#include "tensor.h"
#include "tensor_impl.h"

#include "context.h"

#include "layer/activation_layer.h"
#include "core/layer.h"

#include "random.h"

#include "log.h"
#include <stdio.h>

void test_tensor()
{
    tensor_shape_t shape = make_tensor_shape(4, 1, 2, 3, 4);
    LOG_INFO("Created shape\n");

    size_t size = tensor_size_from_shape(&shape);
    LOG_INFO("Tensor size is %zu\n", size);

    tensor_t tensor; 
    tensor_allocate(&tensor, &shape);
    LOG_INFO("Allocated tensor\n");

    LOG_INFO("Tensor data:\n");
    float* data = tensor_get_data(&tensor);
    for (size_t i = 0; i < size; i++) {
        printf("%f ", data[i]);
    }
    printf("\n");

    LOG_INFO("Filling tensor\n");
    tensor_fill(&tensor, 42.0f);

    LOG_INFO("Tensor data:\n");
    for (size_t i = 0; i < size; i++) {
        printf("%f ", data[i]);
    }
    printf("\n");

    LOG_INFO("Destroying tensor\n");
    tensor_destory(&tensor);
}


void test_activation_layer()
{
    layer_t layer;

    tensor_shape_t input_shape = make_tensor_shape(4, 1, 1, 2, 2);
    size_t size = tensor_size_from_shape(&input_shape);

    activation_layer_create_info_t act_config = {
        .activation_function = ACTIVATION_FUNCTION_RELU
    };

    LOG_INFO("Creating activation layer\n");
    layer_create(&layer, &activation_layer_impl, &act_config, &input_shape, 0);

    LOG_INFO("Allocating input tensor\n");
    tensor_t input;
    tensor_allocate(&input, &input_shape);
    float* input_data = tensor_get_data(&input);
    for (size_t i = 0; i < size; i++) {
        input_data[i] = RandomNormal(0.0f, 1.0f);
        printf("%f ", input_data[i]);
    }
    printf("\n");

    LOG_INFO("Applying activation\n");
    tensor_t* output;
    layer_forward(layer, LAYER_FORWARD_TRAINING, &input, &output);

    float* output_data = tensor_get_data(output);
    for (size_t i = 0; i < size; i++) {
        printf("%f ", output_data[i]);
    }
    printf("\n");

    LOG_INFO("Destroying input\n");
    tensor_destory(&input);
    LOG_INFO("Destroying layer\n");
    layer_destroy(layer);
}


int main() {
    
    if (backend_context_init() != 0) {
        LOG_ERROR("Backend context init failed\n");
    }
    LOG_INFO("Initialized backend context\n");

    test_activation_layer();
}
