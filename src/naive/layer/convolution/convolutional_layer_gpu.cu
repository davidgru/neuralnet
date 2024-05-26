#include "_cuda.h"

#include "convolutional_layer_internal.h"
extern "C" {
#include "tensor/tensor_math.h"
}


__global__
void conv2d_kernel(const float* input, const float* kernel, float* output, int input_height,
    int input_width, int output_height, int output_width, int kernel_height, int kernel_width,
    int stride_y, int stride_x, int padding_y, int padding_x, int dilation_y, int dilation_x,
    int skip_output_y, int skip_output_x, int flip_kernel)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = idx / output_width;
    const int c = idx % output_width;

    if (r < output_height && c < output_width) {        
        const int data_r = (r + skip_output_y) * stride_y - padding_y;
        const int data_c = (c + skip_output_x) * stride_x - padding_x;

        /* calculate the bounds of the kernel to skip the elements that are in the padding */
        const int kr_start = max(0, div_ceil(-data_r, dilation_y));
        const int kr_end = min(kernel_height, div_ceil(input_height - data_r, dilation_y));
        const int kc_start = max(0, div_ceil(-data_c, dilation_x));
        const int kc_end = min(kernel_width, div_ceil(input_width - data_c, dilation_x));

        float sum = 0.0;
        for (int kr = kr_start; kr < kr_end; kr++) { 
            for (int kc = kc_start; kc < kc_end; kc++) {
                const int data_rk = data_r + kr * dilation_y;
                const int data_ck = data_c + kc * dilation_x;
                const int kernel_idx = flip_kernel ?
                    ((kernel_height - kr - 1) * kernel_width + (kernel_width - kc - 1)) :
                    (kr * kernel_width + kc);
                sum += input[data_rk * input_width + data_ck]
                    * kernel[kernel_idx];
            }
        }
        output[r * output_width + c] += sum;
    }
}


__global__
void convolution_forward_kernel(const float* input, const float* filter, const float* bias, float* output,
    int batch_size, int output_channels, int input_channels, int input_height, int input_width,
    int filter_height, int filter_width, int output_height, int output_width, int stride_y, int stride_x,
    int padding_y, int padding_x, int dilation_y, int dilation_x, int skip_output_y, int skip_output_x,
    int flip_kernel)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int column = batch_idx % output_width; batch_idx /= output_width;
    const int row = batch_idx % output_height; batch_idx /= output_height;
    const int oc_idx = batch_idx % output_channels; batch_idx /= output_channels;

    const int input_row = (row + skip_output_y) * stride_y - padding_y;
    const int input_column = (column + skip_output_x) * stride_x - padding_x;

    input += batch_idx * input_channels * input_height * input_width;
    filter += oc_idx * input_channels * filter_height * filter_width;
    output += (batch_idx * output_channels + oc_idx) * output_height * output_width;

    if (batch_idx < batch_size) {
        float sum = 0.0f;
        for (int ic_idx = 0; ic_idx < input_channels; ic_idx++) {
            for (int kr = 0; kr < filter_height; kr++) {
                for (int kc = 0; kc < filter_width; kc++) {
                    const int data_rk = input_row + kr * dilation_y;
                    const int data_ck = input_column + kc * dilation_x;
                    const int kernel_idx = flip_kernel ?
                        ((filter_height - kr - 1) * filter_width + (filter_width - kc - 1)) :
                        (kr * filter_width + kc);
                    if (data_rk >= 0 && data_rk < input_height && data_ck >= 0 && data_ck < input_width) {
                        sum += input[data_rk * input_width + data_ck] * filter[kernel_idx];
                    }
                }
            }
            input += input_height * input_width;
            filter += filter_height * filter_width;
        }
        output[row * output_width + column] += sum + bias[oc_idx];
    }
}


__global__
void convolution_backward_data_kernel(const float* prev_grad, const float* filter, float* grad,
    int batch_size, int prev_grad_height, int prev_grad_width, int in_channels, int out_channels,
    int filter_height, int filter_width, int grad_height, int grad_width, int stride_y, int stride_x,
    int padding_y, int padding_x)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int column = batch_idx % grad_width; batch_idx /= grad_width;
    const int row = batch_idx % grad_height; batch_idx /= grad_height;
    const int ic_idx = batch_idx % out_channels; batch_idx /= out_channels;

    const int prev_grad_row = row + padding_y - filter_height + 1;
    const int prev_grad_column = column + padding_x - filter_width + 1;

    prev_grad += batch_idx * out_channels * prev_grad_height * prev_grad_width;
    filter += ic_idx * filter_height * filter_width;
    grad += (batch_idx * in_channels + ic_idx) * grad_height * grad_width;
    if (batch_idx < batch_size) {
        float sum = 0.0f;
        for (int oc_idx = 0; oc_idx < out_channels; oc_idx++) {
            for (int kr = 0; kr < filter_height; kr++) {
                for (int kc = 0; kc < filter_width; kc++) {
                    const int prev_grad_rk = prev_grad_row + kr * stride_y;
                    const int prev_grad_ck = prev_grad_column + kc * stride_x;
                    const int kernel_idx = (filter_height - kr - 1) * filter_width
                                            + (filter_width - kc - 1);
                    if (prev_grad_rk >= 0 && prev_grad_rk < prev_grad_height
                        && prev_grad_ck >= 0 && prev_grad_ck < prev_grad_width) {
                        sum += prev_grad[prev_grad_rk * prev_grad_width + prev_grad_ck]
                                * filter[kernel_idx];
                    }
                }
            }
            prev_grad += prev_grad_height * prev_grad_width;
            filter += filter_height * filter_width;
        }
        grad[row * grad_width + column] += sum;
    }
}


__global__
void convolution_backward_weights_kernel_parallel_batch(const float* input, const float* prev_grad, float* d_filter,
    int batch_size, int in_channels, int input_height, int input_width, int prev_grad_channels,
    int prev_grad_height, int prev_grad_width, int filter_height, int filter_width, int stride_y,
    int stride_x, int padding_y, int padding_x)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int kcol = batch_idx % filter_width; batch_idx /= filter_width;
    const int krow = batch_idx % filter_height; batch_idx /= filter_height;
    const int ic_idx = batch_idx % in_channels; batch_idx /= in_channels;
    const int oc_idx = batch_idx % prev_grad_channels; batch_idx /= prev_grad_channels;

    const int input_row = krow - padding_y;
    const int input_column = kcol - padding_x;        

    input += (batch_idx * in_channels + ic_idx) * input_height * input_width;
    prev_grad += (batch_idx * prev_grad_channels + oc_idx) * prev_grad_height * prev_grad_width;
    d_filter += ((batch_idx * prev_grad_channels + oc_idx) * in_channels + ic_idx) * filter_height * filter_width;

    if (batch_idx < batch_size) {
        float sum = 0.0f;
        for (int gr = 0; gr < prev_grad_height; gr++) {
            for (int gc = 0; gc < prev_grad_width; gc++) {
                const int data_rk = input_row + gr * stride_y;
                const int data_ck = input_column + gc * stride_x;
                if (data_rk >= 0 && data_rk < input_height
                    && data_ck >= 0 && data_ck < input_width) {
                    sum += input[data_rk * input_width + data_ck]
                         * prev_grad[gr * prev_grad_width + gc];
                }
            }
        }
        d_filter[krow * filter_width + kcol] += sum;
    }
}


void conv2d_gpu(const float* input, const float* kernel, float* output, int32_t input_height,
    int32_t input_width, int32_t kernel_height, int32_t kernel_width, int32_t stride_y,
    int32_t stride_x, int32_t padding_y, int32_t padding_x, int32_t dilation_y, int32_t dilation_x,
    int32_t skip_output_y, int32_t skip_output_x, int32_t flip_kernel)
{
    const int output_height = conv_output_size(input_height, kernel_height, stride_y, dilation_y,
        padding_y) - 2 * skip_output_y;
    const int output_width = conv_output_size(input_width, kernel_width, stride_x, dilation_x,
        padding_x) - 2 * skip_output_x;

    const cuda_props_t* props = get_cuda_props();
    const int block_size = props->default_block_size_1d.x;
    const int num_blocks = cuda_calc_num_blocks((output_height * output_width), block_size);

    conv2d_kernel<<<num_blocks, block_size>>>(input, kernel, output, input_height, input_width, output_height,
        output_width, kernel_height, kernel_width, stride_y, stride_x, padding_y, padding_x, dilation_y,
        dilation_x, skip_output_y, skip_output_x, flip_kernel);
    CUDA_CHECK_LAST_ERROR();
}


void convolution_forward_gpu(const tensor_t* input, const tensor_t* filter, const tensor_t* bias,
    tensor_t* output, int32_t stride_y, int32_t stride_x, int32_t padding_y, int32_t padding_x,
    int32_t dilation_y, int32_t dilation_x, int32_t skip_output_y, int32_t skip_output_x,
    int32_t flip_kernel)
{
    const unsigned int num_threads = tensor_batch_size(output) * tensor_channels(output)
        * tensor_height(output) * tensor_width(output);

    const cuda_props_t* props = get_cuda_props();
    const dim3 block_size = props->default_block_size_1d;
    const dim3 block_dim = {
        cuda_calc_num_blocks(num_threads, block_size.x), 1, 1
    };

    convolution_forward_kernel<<<block_dim, block_size>>>(input->data, filter->data, bias->data, output->data,
        tensor_batch_size(input), tensor_channels(output), tensor_channels(input), tensor_height(input),
        tensor_width(input), _filter_height(filter), _filter_width(filter), tensor_height(output), tensor_width(output),
        stride_y, stride_x, padding_y, padding_x, dilation_y, dilation_x, skip_output_y, skip_output_x, flip_kernel);
    CUDA_CHECK_LAST_ERROR();
}


void convolution_backward_data_gpu(const tensor_t* prev_grad, const tensor_t* filter, tensor_t* grad,
    int32_t stride_y, int32_t stride_x, int32_t padding_y, int32_t padding_x)
{
    const unsigned int num_threads = tensor_batch_size(grad) * tensor_channels(grad)
        * tensor_height(grad) * tensor_width(grad);

    const cuda_props_t* props = get_cuda_props();
    const dim3 block_size = props->default_block_size_1d;
    const dim3 block_dim = {
        cuda_calc_num_blocks(num_threads, block_size.x), 1, 1
    };

    convolution_backward_data_kernel<<<block_dim, block_size>>>(prev_grad->data, filter->data, grad->data,
        tensor_batch_size(prev_grad), tensor_height(prev_grad), tensor_width(prev_grad), tensor_channels(grad),
        tensor_channels(prev_grad), _filter_height(filter), _filter_width(filter), tensor_height(grad),
        tensor_width(grad), stride_y, stride_x, padding_y, padding_x);
    CUDA_CHECK_LAST_ERROR();
}


void convolution_backward_weights_gpu(const tensor_t* input, const tensor_t* prev_grad, tensor_t* d_weights,
    tensor_t* d_bias, int32_t stride_y, int32_t stride_x, int32_t padding_y, int32_t padding_x)
{
    tensor_t d_weights_per_batch;
    tensor_shape_t d_weights_per_batch_shape = make_tensor_shape(5,
        tensor_batch_size(input),
        tensor_channels(prev_grad),
        tensor_channels(input),
        _filter_height(d_weights),
        _filter_width(d_weights)
    );
    tensor_allocate_device(&d_weights_per_batch, &d_weights_per_batch_shape, device_gpu);
    tensor_set_zero(&d_weights_per_batch);

    const unsigned int d_weights_num_threads = tensor_batch_size(input) * tensor_channels(prev_grad) * tensor_channels(input) 
        * _filter_height(d_weights) * _filter_width(d_weights);

    const cuda_props_t* props = get_cuda_props();
    const dim3 block_size = props->default_block_size_1d;
    const dim3 block_dim = {
        cuda_calc_num_blocks(d_weights_num_threads, block_size.x), 1, 1
    };

    /* compute d_weights for each batch in parallel */
    convolution_backward_weights_kernel_parallel_batch<<<block_dim, block_size>>>(input->data, prev_grad->data,
        d_weights_per_batch.data, tensor_batch_size(input), tensor_channels(input), tensor_height(input), tensor_width(input),
        tensor_channels(prev_grad), tensor_height(prev_grad), tensor_width(prev_grad), _filter_height(d_weights),
        _filter_width(d_weights), stride_y, stride_x, padding_y, padding_x);
    CUDA_CHECK_LAST_ERROR();

    /* sum d_weights along batch dimension */
    tensor_sum_axis(d_weights, &d_weights_per_batch, 0);

    /* d_bias is simply a sum of the previous gradient */
    /* for lack of better suiting function, first reduce width and height dim and then reduce batch dim */
    tensor_t prev_grad_view_flattened_wh = {
        .shape = make_tensor_shape(3,
            tensor_batch_size(prev_grad),
            tensor_channels(prev_grad),
            tensor_per_channel_size(prev_grad)
        ),
        .device = device_gpu,
        .data = prev_grad->data
    };

    /* reduce height and width */
    tensor_t d_bias_per_batch;
    tensor_shape_t prev_reduced_wh_shape = make_tensor_shape(2,
        tensor_batch_size(prev_grad),
        tensor_channels(prev_grad)
    );
    tensor_allocate_device(&d_bias_per_batch, &prev_reduced_wh_shape, device_gpu);
    tensor_set_zero(&d_bias_per_batch);

    tensor_sum_axis(&d_bias_per_batch, &prev_grad_view_flattened_wh, 2);
    /* reduce batch dimension */
    tensor_sum_axis(d_bias, &d_bias_per_batch, 0);

    tensor_destory(&d_weights_per_batch);
    tensor_destory(&d_bias_per_batch);
}


#if 0 /* slower because batch is handled sequentially but left for reference */
__global__
void convolution_backward_weights_kernel(const float* input, const float* prev_grad, float* d_filter,
    int batch_size, int in_channels, int input_height, int input_width, int prev_grad_channels,
    int prev_grad_height, int prev_grad_width, int filter_height, int filter_width, int stride_y,
    int stride_x, int padding_y, int padding_x)
{
    int oc_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int kcol = oc_idx % filter_width; oc_idx /= filter_width;
    const int krow = oc_idx % filter_height; oc_idx /= filter_height;
    const int ic_idx = oc_idx % in_channels; oc_idx /= in_channels;

    const int input_row = krow - padding_y;
    const int input_column = kcol - padding_x;        

    input += ic_idx * input_height * input_width;
    prev_grad += oc_idx * prev_grad_height * prev_grad_width;
    d_filter += (oc_idx * in_channels + ic_idx) * filter_height * filter_width;

    if (oc_idx < prev_grad_channels) {
        float sum = 0.0f;
        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            for (int gr = 0; gr < prev_grad_height; gr++) {
                for (int gc = 0; gc < prev_grad_width; gc++) {
                    const int data_rk = input_row + gr * stride_y;
                    const int data_ck = input_column + gc * stride_x;
                    if (data_rk >= 0 && data_rk < input_height
                        && data_ck >= 0 && data_ck < input_width) {
                        sum += input[data_rk * input_width + data_ck]
                             * prev_grad[gr * prev_grad_width + gc];
                    }
                }
            }
            input += in_channels * input_height * input_width;
            prev_grad += prev_grad_channels * prev_grad_height * prev_grad_width;
        }
        d_filter[krow * filter_width + kcol] += sum;
    }
}


__global__
void convolution_backward_bias_kernel(const float* prev_grad, float* d_bias, int batch_size,
    int prev_grad_channels, int prev_grad_height, int prev_grad_width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < prev_grad_channels) {
        float sum = 0.0f;
        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            for (int gr = 0; gr < prev_grad_height; gr++) {
                for (int gc = 0; gc < prev_grad_width; gc++) {
                    sum += prev_grad[((batch_idx * prev_grad_channels + idx)
                            * prev_grad_height + gr) * prev_grad_width + gc];
                }
            }
        }
        d_bias[idx] += sum;
    }
}


void convolution_backward_weights_gpu(const tensor_t* input, const tensor_t* prev_grad, tensor_t* d_weights,
    tensor_t* d_bias, int32_t stride_y, int32_t stride_x, int32_t padding_y, int32_t padding_x)
{
    const unsigned int num_threads = tensor_channels(prev_grad) * tensor_channels(d_weights) 
        * _filter_height(d_weights) * _filter_width(d_weights);

    const cuda_props_t* props = get_cuda_props();
    const dim3 block_size = props->default_block_size_1d;
    const dim3 block_dim = {
        cuda_calc_num_blocks(num_threads, block_size.x), 1, 1
    };

    convolution_backward_weights_kernel<<<block_dim, block_size>>>(input->data, prev_grad->data, d_weights->data,
        tensor_batch_size(input), tensor_channels(input), tensor_height(input), tensor_width(input),
        tensor_channels(prev_grad), tensor_height(prev_grad), tensor_width(prev_grad), _filter_height(d_weights),
        _filter_width(d_weights), stride_y, stride_x, padding_y, padding_x);
    CUDA_CHECK_LAST_ERROR();

    const dim3 bias_bs = props->default_block_size_1d;
    const dim3 bias_nb = {
        cuda_calc_num_blocks(tensor_channels(prev_grad), bias_bs.x)
    };

    convolution_backward_bias_kernel<<<bias_nb, bias_nb>>>(prev_grad->data, d_bias->data, tensor_batch_size(prev_grad),
        tensor_channels(prev_grad), tensor_height(prev_grad), tensor_width(prev_grad));
    CUDA_CHECK_LAST_ERROR();
}
#endif
