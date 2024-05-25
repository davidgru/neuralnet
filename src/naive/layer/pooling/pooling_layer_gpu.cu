#include "_cuda.h"
#include "pooling_layer_internal.h"


#define _max(x,y) ((x) > (y) ? (x) : (y))
#define _min(x,y) ((x) < (y) ? (x) : (y))


typedef float (*pool_fwd_func_t)(const float*, int, int, int);
typedef void (*pool_bwd_func_t)(const float*, float*, float, int, int, int);


__device__ float pool_op_avg(const float* input, int input_stride, int kernel_height, int kernel_width)
{
    float res = 0.0f;
    for (int kr = 0; kr < kernel_height; kr++) {
        for (int kc = 0; kc < kernel_width; kc++) {
            res += input[kr * input_stride + kc];
        }
    }
    return res / (kernel_height * kernel_width);
}

__device__ float pool_op_max(const float* input, int input_stride, int kernel_height, int kernel_width)
{
    float res = input[0];
    for (int kr = 0; kr < kernel_height; kr++) {
        for (int kc = 0; kc < kernel_width; kc++) {
            res = _max(res, input[kr * input_stride + kc]);
        }
    }
    return res;
}

__device__ float pool_op_min(const float* input, int input_stride, int kernel_height, int kernel_width)
{
    float res = input[0];
    for (int kr = 0; kr < kernel_height; kr++) {
        for (int kc = 0; kc < kernel_width; kc++) {
            res = _min(res, input[kr * input_stride + kc]);
        }
    }
    return res;
}

__device__ void pool_bwd_avg(const float* input, float* grad, float prev_grad, int input_stride,
                             int kernel_height, int kernel_width)
{
    for (int kr = 0; kr < kernel_height; kr++) {
        for (int kc = 0; kc < kernel_width; kc++) {
            grad[kr * input_stride + kc] += prev_grad / (kernel_height * kernel_width);
        }
    }
}

__device__ void pool_bwd_max(const float* input, float* grad, float prev_grad, int input_stride,
                             int kernel_height, int kernel_width)
{
    float _max = input[0];
    int max_idx = 0;
    for (int kr = 0; kr < kernel_height; kr++) {
        for (int kc = 0; kc < kernel_width; kc++) {
            int curr_idx = kr * input_stride + kc;
            if (input[curr_idx] > _max) {
                max_idx = curr_idx;
                _max = input[curr_idx];
            }
        }
    }
    grad[max_idx] += prev_grad;
}

__device__ void pool_bwd_min(const float* input, float* grad, float prev_grad, int input_stride,
                             int kernel_height, int kernel_width)
{
    float _min = input[0];
    int min_idx = 0;
    for (int kr = 0; kr < kernel_height; kr++) {
        for (int kc = 0; kc < kernel_width; kc++) {
            int curr_idx = kr * input_stride + kc;
            if (input[curr_idx] < _min) {
                min_idx = curr_idx;
                _min = input[curr_idx];
            }
        }
    }
    grad[min_idx] += prev_grad;
}

template<pool_fwd_func_t POOL_OP>
__global__ void pooling_forward_kernel(const float* input, float* output, int batch_size,
    int channels, int output_height, int output_width, int kernel_height, int kernel_width)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = batch_idx % output_width; batch_idx /= output_width;
    const int row = batch_idx % output_height; batch_idx /= output_height;
    const int channel = batch_idx % channels; batch_idx /= channels;

    const int input_height = kernel_height * output_height;
    const int input_width = kernel_width * output_width;
    const int input_row = kernel_height * row;
    const int input_col = kernel_width * col;

    input += ((batch_idx * channels + channel) * input_height + input_row) * input_width + input_col;
    output += ((batch_idx * channels + channel) * output_height + row) * output_width + col;

    if (batch_idx < batch_size) {
        *output = POOL_OP(input, input_width, kernel_height, kernel_width);
    }
}

template<pool_bwd_func_t POOL_BWD_OP>
__global__ void pooling_backward_kernel(const float* input, const float* prev_grad, float* grad,
    int batch_size, int channels, int prev_grad_height, int prev_grad_width, int kernel_height,
    int kernel_width)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = batch_idx % prev_grad_width; batch_idx /= prev_grad_width;
    const int row = batch_idx % prev_grad_height; batch_idx /= prev_grad_height;
    const int channel = batch_idx % channels; batch_idx /= channels;

    const int input_height = kernel_height * prev_grad_height;
    const int input_width = kernel_width * prev_grad_width;
    const int input_row = kernel_height * row;
    const int input_col = kernel_width * col;

    input += ((batch_idx * channels + channel) * input_height + input_row) * input_width + input_col;
    grad += ((batch_idx * channels + channel) * input_height + input_row) * input_width + input_col;
    prev_grad += ((batch_idx * channels + channel) * prev_grad_height + row) * prev_grad_width + col;

    if (batch_idx < batch_size) {
        POOL_BWD_OP(input, grad, *prev_grad, input_width, kernel_height, kernel_width);
    }
}


void pooling_forward_gpu(const tensor_t* input, tensor_t* output, size_t kernel_width, pooling_kind_t kind)
{
    const int num_threads = tensor_batch_size(output) * tensor_channels(output)
                           * tensor_height(output) * tensor_width(output);

    const cuda_props_t* props = get_cuda_props();
    const unsigned int block_size = props->default_block_size_1d.x;
    const unsigned int num_blocks = cuda_calc_num_blocks(num_threads, block_size);

    switch(kind) {
        case POOLING_AVERAGE:
            pooling_forward_kernel<pool_op_avg><<<num_blocks, block_size>>>(input->data, output->data,
                tensor_batch_size(output), tensor_channels(output), tensor_height(output), tensor_width(output),
                kernel_width, kernel_width);
            break;
        case POOLING_MAX:
            pooling_forward_kernel<pool_op_max><<<num_blocks, block_size>>>(input->data, output->data,
                tensor_batch_size(output), tensor_channels(output), tensor_height(output), tensor_width(output),
                kernel_width, kernel_width);
            break;
        case POOLING_MIN:
            pooling_forward_kernel<pool_op_min><<<num_blocks, block_size>>>(input->data, output->data,
                tensor_batch_size(output), tensor_channels(output), tensor_height(output), tensor_width(output),
                kernel_width, kernel_width);
            break;
    }
}

void pooling_backward_gpu(const tensor_t* input, const tensor_t* prev_grad, tensor_t* grad, size_t kernel_width, pooling_kind_t kind)
{
    const int num_threads = tensor_batch_size(prev_grad) * tensor_channels(prev_grad)
                           * tensor_height(prev_grad) * tensor_width(prev_grad);

    const cuda_props_t* props = get_cuda_props();
    const unsigned int block_size = props->default_block_size_1d.x;
    const unsigned int num_blocks = cuda_calc_num_blocks(num_threads, block_size);

    switch(kind) {
        case POOLING_AVERAGE:
            pooling_backward_kernel<pool_bwd_avg><<<num_blocks, block_size>>>(input->data, prev_grad->data,
                grad->data, tensor_batch_size(prev_grad), tensor_channels(prev_grad), tensor_height(prev_grad),
                tensor_width(prev_grad), kernel_width, kernel_width);
            break;
        case POOLING_MAX:
            pooling_backward_kernel<pool_bwd_max><<<num_blocks, block_size>>>(input->data, prev_grad->data,
                grad->data, tensor_batch_size(prev_grad), tensor_channels(prev_grad), tensor_height(prev_grad),
                tensor_width(prev_grad), kernel_width, kernel_width);
            break;
        case POOLING_MIN:
            pooling_backward_kernel<pool_bwd_min><<<num_blocks, block_size>>>(input->data, prev_grad->data,
                grad->data, tensor_batch_size(prev_grad), tensor_channels(prev_grad), tensor_height(prev_grad),
                tensor_width(prev_grad), kernel_width, kernel_width);
            break;
    }
}
