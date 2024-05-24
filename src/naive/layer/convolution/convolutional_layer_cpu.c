#include "convolutional_layer_internal.h"

#include "util/ai_math.h"
#include "tensor/tensor_impl.h"


void conv2d_cpu(const float* input, const float* kernel, float* output, int32_t input_height,
    int32_t input_width, int32_t kernel_height, int32_t kernel_width, int32_t stride_y,
    int32_t stride_x, int32_t padding_y, int32_t padding_x, int32_t dilation_y, int32_t dilation_x,
    int32_t skip_output_y, int32_t skip_output_x, int32_t flip_kernel)
{
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

                    int32_t kernel_idx = flip_kernel ?
                        ((kernel_height - kr - 1) * kernel_width + (kernel_width - kc - 1)) :
                        (kr * kernel_width + kc);
            
                    output[r * output_width + c] += input[data_rk * input_width + data_ck]
                        * kernel[kernel_idx];
                }
            }
        }
    }
}


void convolution_forward_cpu(const tensor_t* input, const tensor_t* filter, const tensor_t* bias,
    tensor_t* output, int32_t stride_y, int32_t stride_x, int32_t padding_y, int32_t padding_x,
    int32_t dilation_y, int32_t dilation_x, int32_t skip_output_y, int32_t skip_output_x,
    int32_t flip_kernel)
{
    const float* x = input->data;
    const float* w = filter->data;
    const float* b = bias->data;
    float* y = output->data;

    for (size_t n = 0; n < tensor_batch_size(input); n++) {
        for (size_t i = 0; i < tensor_channels(output); i++) {
            float* _y = y + n * tensor_per_batch_size(output) + i * tensor_per_channel_size(output);
            for (size_t j = 0; j < tensor_channels(input); j++) {
                const float* _x = x + n * tensor_per_batch_size(input) + j * tensor_per_channel_size(input);
                const float* _w = w + i * _filter_size(filter) + j * _filter_width(filter) * _filter_height(filter);
                // Do a convolution with the one input channel and one filter channel to produce part of one output feature map.
                conv2d_cpu(_x, _w, _y, tensor_height(input), tensor_width(input), _filter_height(filter), _filter_width(filter),
                    stride_y, stride_x, padding_y,
                    padding_x, 1, 1, 0, 0, false);
            }
            // Add the bias to every element of the feature map
            VectorAddScalar(_y, b[i], tensor_per_channel_size(output));
        }
    }
}


void convolution_backward_data_cpu(const tensor_t* prev_grad, const tensor_t* filter, tensor_t* grad,
    int32_t stride_y, int32_t stride_x, int32_t padding_y, int32_t padding_x)
{
    const float* dy = prev_grad->data;
    const float* w = filter->data;
    float* dx = grad->data;

    for (size_t n = 0; n < tensor_batch_size(prev_grad); n++) {
        for (size_t i = 0; i < tensor_channels(grad); i++) {
            float* _dx = dx + n * tensor_per_batch_size(grad) * tensor_channels(grad) + i * tensor_per_channel_size(grad);
            for (size_t j = 0; j < tensor_channels(prev_grad); j++) {
                const float* _dy = dy + n * tensor_per_batch_size(prev_grad) + j * tensor_per_channel_size(prev_grad);
                const float* _w = w + j + _filter_size(filter) + i * _filter_height(filter) * _filter_width(filter);
                /* dx = conv2d(w, flip(dy),
                            dilation: (stride_y,stride_x),
                            padding: (output_height-1,output_width-1))
                    = conv2d(dy, flip(w),
                            dilation: (stride_y, stride_x),
                            padding: (filter_height-1,filter_width-1)) */
                conv2d_cpu(_dy, _w, _dx, tensor_height(prev_grad), tensor_width(prev_grad), _filter_height(filter), _filter_width(filter),
                        1, 1, _filter_height(filter) - 1, _filter_width(filter) - 1, stride_y,
                        stride_x, padding_y, padding_x,
                        true);
            }
        }
    }
}
