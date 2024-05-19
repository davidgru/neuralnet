
#include "weight_init.h"

#include "random.h"

#include <math.h>


static void winit_get_input_output_size(const tensor_shape_t* shape, size_t* out_input_size, size_t* out_output_size)
{
    *out_output_size = tensor_shape_get_dim(shape, 0);
    *out_input_size = tensor_size_from_shape(shape) / *out_output_size;
}


static void winit_xavier_raw(float* data, size_t data_size, size_t input_size, size_t output_size)
{
    for (size_t i = 0; i < data_size; i++) {
        data[i] = RandomNormal(0.0f, sqrtf(1.0f / (input_size + output_size)));
    }
}


static void winit_he_raw(float* data, size_t data_size, size_t input_size, size_t output_size)
{
    for (size_t i = 0; i < data_size; i++) {
        data[i] = RandomNormal(0.0f, sqrtf(2.0f / (input_size + output_size)));
    }
}


void winit_xavier(tensor_t* tensor)
{
    const tensor_shape_t* shape = tensor_get_shape(tensor);
    const size_t data_size = tensor_size_from_shape(shape);
    float* data = tensor_get_data(tensor);

    size_t input_size = 0;
    size_t output_size = 0;
    winit_get_input_output_size(shape, &input_size, &output_size);

    winit_xavier_raw(data, data_size, input_size, output_size);
}


void winit_he(tensor_t* tensor)
{
    const tensor_shape_t* shape = tensor_get_shape(tensor);
    const size_t data_size = tensor_size_from_shape(shape);
    float* data = tensor_get_data(tensor);

    size_t input_size = 0;
    size_t output_size = 0;
    winit_get_input_output_size(shape, &input_size, &output_size);
    
    winit_he_raw(data, data_size, input_size, output_size);
}


void winit_zeros(tensor_t* tensor)
{
    tensor_set_zero(tensor);
}
