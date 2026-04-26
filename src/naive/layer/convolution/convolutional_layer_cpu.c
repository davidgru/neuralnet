#include "convolutional_layer_internal.h"

#include "util/ai_math.h"
#include "tensor/tensor_impl.h"
#include "tensor/tensor_math.h"

#if defined(USE_AVX)

#include <malloc.h>

#include "simd/simd.h"

void convolution_forward_cpu(const tensor_t* input, const tensor_t* filter, const tensor_t* bias,
    tensor_t* output, int32_t stride_y, int32_t stride_x, int32_t padding_y, int32_t padding_x,
    int32_t dilation_y, int32_t dilation_x)
{
    /* Conv */
    conv(input->data, filter->data, output->data,
         tensor_batch_size(input), tensor_channels(input), tensor_height(input), tensor_width(input),
         tensor_channels(output), _filter_height(filter), _filter_width(filter),
         padding_y, padding_x, stride_y, stride_x, dilation_y, dilation_x, 0, 0);

    /* Bias */
    for (int32_t n = 0; n < tensor_batch_size(output); ++n)
    for (int32_t oc = 0; oc < tensor_channels(output); ++oc)
    {
        vec_scalar_add(
            &output->data[
                 n * tensor_per_batch_size(output)
              + oc * tensor_per_channel_size(output)
            ],
            bias->data[oc],
            tensor_per_channel_size(output)
        );
    }
}

void convolution_backward_data_cpu(const tensor_t* prev_grad, const tensor_t* filter, tensor_t* grad,
    int32_t stride_y, int32_t stride_x, int32_t padding_y, int32_t padding_x)
{
    const float* dy = prev_grad->data;
    const float* w = filter->data;
    float* dx = grad->data;

    const size_t N = tensor_batch_size(grad);
    const size_t IC = tensor_channels(grad);
    const size_t IH = tensor_height(grad);
    const size_t IW = tensor_width(grad);
    const size_t OC = tensor_channels(prev_grad);
    const size_t OH = tensor_height(prev_grad);
    const size_t OW = tensor_width(prev_grad);
    const size_t KH = _filter_height(filter);
    const size_t KW = _filter_width(filter);
    
    /* dx = conv2d(w, flip(dy),
                   dilation: (stride_y,stride_x),
                   padding: (output_height-1,output_width-1))
          = conv2d(dy, flip(transpose(w)),
                   dilation: (stride_y, stride_x),
                   padding: (filter_height-1,filter_width-1)) */

    // permute W"[ic,oc,kh,kw] = W[oc,ic,KH-kh-1,KW-kw-1]  (flip and transpose)
    float* w_p = malloc(OC * IC * KH * KW * sizeof(float));
    
    for (size_t oc = 0; oc < OC; ++oc)
    for (size_t ic = 0; ic < IC; ++ic)
    for (size_t kh = 0; kh < KH; ++kh)
    for (size_t kw = 0; kw < KW; ++kw)
    {
        w_p[
            ic * (OC * KH * KW)
          + oc * (KH * KW)
          + (KH - kh - 1) * (KW)
          + (KW - kw - 1)
        ] = w[
            oc * (IC * KH * KW)
          + ic * (KH * KW)
          + kh * (KW)
          + kw            
        ];
    }

    conv(dy, w_p, dx, N, OC, OH, OW, IC, KH, KW,
         KH - 1, KW - 1, 1, 1, stride_y, stride_x,
         padding_y, padding_x);

    free(w_p);


    // for (size_t n = 0; n < tensor_batch_size(prev_grad); n++) {
    //     for (size_t i = 0; i < tensor_channels(grad); i++) {
    //         float* _dx = dx + n * tensor_per_batch_size(grad) + i * tensor_per_channel_size(grad);
    //         for (size_t j = 0; j < tensor_channels(prev_grad); j++) {
    //             const float* _dy = dy + n * tensor_per_batch_size(prev_grad) + j * tensor_per_channel_size(prev_grad);
    //             const float* _w = w + j + _filter_size(filter) + i * _filter_height(filter) * _filter_width(filter);
    //             /* dx = conv2d(w, flip(dy),
    //                         dilation: (stride_y,stride_x),
    //                         padding: (output_height-1,output_width-1))
    //                 = conv2d(dy, flip(w),
    //                         dilation: (stride_y, stride_x),
    //                         padding: (filter_height-1,filter_width-1)) */
    //             conv2d_cpu(_dy, _w, _dx, tensor_height(prev_grad), tensor_width(prev_grad), _filter_height(filter), _filter_width(filter),
    //                     1, 1, _filter_height(filter) - 1, _filter_width(filter) - 1, stride_y,
    //                     stride_x, padding_y, padding_x,
    //                     true);
    //         }
    //     }
    // }
}

#include <stdio.h>
void convolution_backward_weights_cpu(const tensor_t* input, const tensor_t* prev_grad, tensor_t* d_weights,
    tensor_t* d_bias, int32_t stride_y, int32_t stride_x, int32_t padding_y, int32_t padding_x)
{
    // dw = conv2d(x, dy, dilation: (stride_y, stride_x))
    
    const float* x = input->data;
    const float* dy = prev_grad->data;
    float* dw = d_weights->data;

    const size_t N = tensor_batch_size(input);
    const size_t IC = tensor_channels(input);
    const size_t IH = tensor_height(input);
    const size_t IW = tensor_width(input);
    const size_t OC = tensor_channels(prev_grad);
    const size_t OH = tensor_height(prev_grad);
    const size_t OW = tensor_width(prev_grad);
    const size_t KH = _filter_height(d_weights);
    const size_t KW = _filter_width(d_weights);

    // actual -> interpreted dim
    // N -> IC
    // OC -> OC
    // IC -> N
    // OH -> KH
    // OW -> KW

    //  x: NxICxIHxIW => (reshape) ICxNxIHxIW ==> (alias) NxICxIHxIW
    // dy: NxOCxOHxOW => (reshape) OCxNxOHxOW ==> (alias) OCxICxKWxKW
    // ==> dw: NxOCxOHxOW ==> (alias) ICxOCxKHxKW => (reshape) OCxICxKHxKW

    float* x_p = malloc(N * IC * IH * IW * sizeof(float));
    float* dy_p = malloc(N * OC * OH * OW * sizeof(float));
    float* dw_p = malloc(OC * IC * KH * KW * sizeof(float));

    // permute x: NxICxIHxIW => ICxNxIHxIW
    swap_N_C(input->data, x_p, N, IC, IH, IW);

    // permute dy: (NxOCxOHxOW) => (OCxNxOHxOW)
    swap_N_C(prev_grad->data, dy_p, N, OC, OH, OW);

    // conv with permuted tensors
    conv(x_p, dy_p, dw_p, IC, N, IH, IW,
         OC, OH, OW, padding_y, padding_x,
         1, 1, stride_y, stride_x, 0, 0);

    // permute dw: ICxOCxKHxKW => OCxICxKHxKW
    swap_N_C(dw_p, d_weights->data, IC, OC, KH, KW);

    free(x_p);
    free(dy_p);
    free(dw_p);

    /* backward bias */
    for (size_t n = 0; n < tensor_batch_size(prev_grad); n++) {
        const float* _dy = dy + n * tensor_per_batch_size(prev_grad);
        for (size_t i = 0; i < tensor_channels(prev_grad); i++) {
            const tensor_t out_channel_grad = {
                .shape = make_tensor_shape(1, tensor_per_channel_size(prev_grad)),
                .data = _dy + i * tensor_per_channel_size(prev_grad),
                .device = device_cpu
            };
            tensor_t db_tensor = {
                .shape = make_tensor_shape(1, 1),
                .data = d_bias->data + i,
                .device = device_cpu
            };
            tensor_sum(&db_tensor, &out_channel_grad);
        }
    }
}

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

#else

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
    int32_t dilation_y, int32_t dilation_x)
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
            float* _dx = dx + n * tensor_per_batch_size(grad) + i * tensor_per_channel_size(grad);
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

void convolution_backward_weights_cpu(const tensor_t* input, const tensor_t* prev_grad, tensor_t* d_weights,
    tensor_t* d_bias, int32_t stride_y, int32_t stride_x, int32_t padding_y, int32_t padding_x)
{
    const float* x = input->data;
    const float* dy = prev_grad->data;
    float* dw = d_weights->data;

    /* backward filter */
    for (size_t n = 0; n < tensor_batch_size(input); n++) {
        for (size_t i = 0; i < tensor_channels(input); i++) {
            const float* _x = x + n * tensor_per_batch_size(input) + i * tensor_per_channel_size(input);
            for (size_t j = 0; j < tensor_channels(prev_grad); j++) {
                const float* _dy = dy + n * tensor_per_batch_size(prev_grad) + j * tensor_per_channel_size(prev_grad);
                float* _dw = dw + j * _filter_size(d_weights) + i * _filter_height(d_weights) * _filter_width(d_weights);
                /* dw = conv2d(x, dy, dilation: (stride_y, stride_x)) */
                conv2d_cpu(_x, _dy, _dw, tensor_height(input), tensor_width(input), tensor_height(prev_grad),
                    tensor_width(prev_grad), 1, 1, padding_y, padding_x, stride_y, stride_x, 0, 0, false);
            }
        }
    }

    /* backward bias */
    for (size_t n = 0; n < tensor_batch_size(prev_grad); n++) {
        const float* _dy = dy + n * tensor_per_batch_size(prev_grad);
        for (size_t i = 0; i < tensor_channels(prev_grad); i++) {
            const tensor_t out_channel_grad = {
                .shape = make_tensor_shape(1, tensor_per_channel_size(prev_grad)),
                .data = _dy + i * tensor_per_channel_size(prev_grad),
                .device = device_cpu
            };
            tensor_t db_tensor = {
                .shape = make_tensor_shape(1, 1),
                .data = d_bias->data + i,
                .device = device_cpu
            };
            tensor_sum(&db_tensor, &out_channel_grad);
        }
    }
}

#endif
