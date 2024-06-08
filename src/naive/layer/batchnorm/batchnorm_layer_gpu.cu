
#include "_cuda.h"
#include "batchnorm_layer_internal.h"
extern "C" {
#include "tensor/tensor.h"
#include "tensor/tensor_math.h"
}

__global__ void batchnorm_forward_kernel(const float* input, const float* mean,
    const float* var, const float* scale, const float* shift, float* output,
    int batch_size, int channels, int per_channel_size, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_idx = (idx / per_channel_size) % channels;
    int batch_idx = ((idx / per_channel_size) / channels);

    if (batch_idx < batch_size) {
        float normed = (input[idx] - mean[channel_idx])
                       / sqrtf(var[channel_idx] + eps);
        output[idx] = scale[channel_idx] * normed + shift[channel_idx];
    }
}


void batchnorm_forward_gpu(const tensor_t* input, const tensor_t* mean, const tensor_t* var,
    const tensor_t* scale, const tensor_t* shift, tensor_t* output, float bn_eps)
{
    const unsigned int sz = tensor_get_size(input);

    const cuda_props_t* props = get_cuda_props();
    const dim3 block_size = props->default_block_size_1d;
    const dim3 block_dim = {
        cuda_calc_num_blocks(sz, block_size.x)
    };

    batchnorm_forward_kernel<<<block_dim, block_size>>>(input->data, mean->data, var->data,
        scale->data, shift->data, output->data, tensor_batch_size(input), tensor_channels(input),
        tensor_per_channel_size(input), bn_eps);
    CUDA_CHECK_LAST_ERROR();
}

__global__ void batchnorm_d_scale_elem_kernel(const float* input, const float* mean,
    const float* var, const float* d_output, float* d_scale_elem,
    int batch_size, int channels, int per_channel_size, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_idx = (idx / per_channel_size) % channels;
    int batch_idx = ((idx / per_channel_size) / channels);

    if (batch_idx < batch_size) {
        float x_hat = (input[idx] - mean[channel_idx])
                       / sqrtf(var[channel_idx] + eps);
        d_scale_elem[idx] = x_hat * d_output[idx];
    }
}

void batchnorm_backward_weights_gpu(const tensor_t* input, const tensor_t* mean, const tensor_t* var,
    const tensor_t* d_output, tensor_t* d_scale, tensor_t* d_shift, float bn_eps, tensor_t* tmp1, tensor_t* tmp2)
{
    const unsigned int sz = tensor_get_size(input);

    const cuda_props_t* props = get_cuda_props();
    const dim3 block_size = props->default_block_size_1d;
    const dim3 block_dim = {
        cuda_calc_num_blocks(sz, block_size.x)
    };

    /* tmp <- d_output * x_hat*/
    batchnorm_d_scale_elem_kernel<<<block_dim, block_size>>>(input->data, mean->data, var->data,
        d_output->data, tmp2->data, tensor_batch_size(input), tensor_channels(input),
        tensor_per_channel_size(input), bn_eps);
    CUDA_CHECK_LAST_ERROR();

    /* accumulating d_scale */
    tensor_t d_scale_elem_view = {
        .shape = make_tensor_shape(3,
            tensor_batch_size(input),
            tensor_channels(input),
            tensor_per_channel_size(input)
        ),
        .device = device_gpu,
        .data = tmp2->data
    };
    tensor_mean_axis(tmp1, &d_scale_elem_view, TENSOR_HEIGHT_DIM);
    tensor_mean_axis(d_scale, tmp1, TENSOR_BATCH_DIM);

    /* accumulating d_shift */
    tensor_t d_output_elem_view = {
        .shape = make_tensor_shape(3,
            tensor_batch_size(input),
            tensor_channels(input),
            tensor_per_channel_size(input)
        ),
        .device = device_gpu,
        .data = d_output->data
    };
    tensor_mean_axis(tmp1, &d_output_elem_view, TENSOR_HEIGHT_DIM);
    tensor_mean_axis(d_shift, tmp1, TENSOR_BATCH_DIM);
}

__global__ void batchnorm_d_x_hat_kernel(const float* d_output, const float* scale, float* d_x_hat,
    int batch_size, int channels, int per_channel_size, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_idx = (idx / per_channel_size) % channels;
    int batch_idx = ((idx / per_channel_size) / channels);

    if (batch_idx < batch_size) {
        d_x_hat[idx] = d_output[idx] * scale[channel_idx];
    }
}

__global__ void batchnorm_d_var_prep_kernel(const float* input, const float* d_x_hat, const float* mean,
    const float* var, float* d_var_elem, int batch_size, int channels, int per_channel_size, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_idx = (idx / per_channel_size) % channels;
    int batch_idx = ((idx / per_channel_size) / channels);

    if (batch_idx < batch_size) {
        d_var_elem[idx] = d_x_hat[idx] * (input[idx] - mean[channel_idx]) 
            * -0.5f * powf(var[channel_idx] + eps, -1.5f);
    }
}

__global__ void batchnorm_d_mean_prep_kernel(const float* input, const float* d_x_hat,
    const float* mean, const float* var, const float* d_var, float* d_mean_elem, int batch_size, int channels,
    int per_channel_size, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_idx = (idx / per_channel_size) % channels;
    int batch_idx = ((idx / per_channel_size) / channels);

    if (batch_idx < batch_size) {
        float acc1 = -d_x_hat[idx] / sqrtf(var[channel_idx] + eps);
        float acc2 = -2.0f * (input[idx] - mean[channel_idx])
            * d_var[channel_idx] / (batch_size * per_channel_size);
        d_mean_elem[idx] = acc1 + acc2;
    }
}

__global__ void batchnorm_d_x_kernel(const float* input, const float* mean, const float* var, const float* d_mean,
    const float* d_var, const float* d_x_hat, float* d_input, int batch_size, int channels, int per_channel_size,
    float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_idx = (idx / per_channel_size) % channels;
    int batch_idx = ((idx / per_channel_size) / channels);

    if (batch_idx < batch_size) {
        d_input[idx] = d_x_hat[idx] / sqrtf(var[channel_idx] + eps)
            + d_var[channel_idx] * 2.0f * (input[idx] - mean[channel_idx]) / (batch_size * per_channel_size)
            + d_mean[channel_idx] / (batch_size * per_channel_size);
    }
}

void batchnorm_backward_data_gpu(const tensor_t* input, const tensor_t* mean, const tensor_t* var,
    const tensor_t* scale, const tensor_t* d_output, tensor_t* tmp_d_mean, tensor_t* tmp_d_var,
    tensor_t* d_input, float bn_eps, tensor_t* tmp1, tensor_t* tmp2)
{
    const unsigned int sz = tensor_get_size(input);

    const cuda_props_t* props = get_cuda_props();
    const dim3 block_size = props->default_block_size_1d;
    const dim3 block_dim = {
        cuda_calc_num_blocks(sz, block_size.x)
    };

    const dim3 per_channel_bs = props->default_block_size_1d;
    const dim3 per_channel_bd = { cuda_calc_num_blocks(tensor_channels(input), per_channel_bs.x) };
    
    /* d_input as temporary storage for d_x_hat */
    batchnorm_d_x_hat_kernel<<<block_dim, block_size>>>(d_output->data, scale->data, d_input->data,
        tensor_batch_size(input), tensor_channels(input), tensor_per_channel_size(input), bn_eps);
    CUDA_CHECK_LAST_ERROR();

    /* calculate d_var*/
    batchnorm_d_var_prep_kernel<<<block_dim, block_size>>>(input->data, d_input->data, mean->data,
        var->data, tmp2->data, tensor_batch_size(input), tensor_channels(input),
        tensor_per_channel_size(input), bn_eps);
    CUDA_CHECK_LAST_ERROR();

    tensor_t d_var_elem_view = {
        .shape = make_tensor_shape(3,
            tensor_batch_size(input),
            tensor_channels(input),
            tensor_per_channel_size(input)
        ),
        .device = device_gpu,
        .data = tmp2->data
    };
    tensor_sum_axis(tmp1, &d_var_elem_view, TENSOR_HEIGHT_DIM);
    tensor_sum_axis(tmp_d_var, tmp1, TENSOR_BATCH_DIM);

    /* calculate d_mean */
    batchnorm_d_mean_prep_kernel<<<block_dim, block_size>>>(input->data, d_input->data, mean->data,
        var->data, tmp_d_var->data, tmp2->data, tensor_batch_size(input), tensor_channels(input),
        tensor_per_channel_size(input), bn_eps);
    CUDA_CHECK_LAST_ERROR();
    tensor_t d_mean_elem_view = {
        .shape = make_tensor_shape(3,
            tensor_batch_size(input),
            tensor_channels(input),
            tensor_per_channel_size(input)
        ),
        .device = device_gpu,
        .data = tmp2->data
    };
    tensor_sum_axis(tmp1, &d_mean_elem_view, TENSOR_HEIGHT_DIM);
    tensor_sum_axis(tmp_d_mean, tmp1, TENSOR_BATCH_DIM);

    /* calculate d_x */
    batchnorm_d_x_kernel<<<block_dim, block_size>>>(input->data, mean->data, var->data, tmp_d_mean->data,
        tmp_d_var->data, d_input->data, d_input->data, tensor_batch_size(input), tensor_channels(input),
        tensor_per_channel_size(input), bn_eps);
    CUDA_CHECK_LAST_ERROR();
}
