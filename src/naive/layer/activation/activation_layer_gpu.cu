#include "_cuda.h"
#include "activation_layer_internal.h"


__global__ void sigmoid_kernel(const float* in, float* out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        out[idx] = 1.0f / (1.0f + expf(-in[idx]));
    }
}

__global__ void tanh_kernel(const float* in, float* out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        out[idx] = tanhf(in[idx]);
    }
}

__global__ void relu_kernel(const float* in, float* out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        out[idx] = in[idx] > 0.0f ? in[idx] : 0.0f;
    }
}

__global__ void dsigmoid_kernel(const float* in, float* out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        out[idx] = (1.0f - in[idx]) * in[idx];
    }
}

__global__ void dtanh_kernel(const float* in, float* out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        out[idx] = 1.0f - (in[idx] * in[idx]);
    }
}

__global__ void drelu_kernel(const float* in, float* out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        out[idx] = in[idx] > 0.0f ? 1.0f : 0.0f;
    }
}


void sigmoid_gpu(const float* in, float* out, size_t size)
{
    const cuda_props_t* props = get_cuda_props();
    const dim3 block_size = props->default_block_size_1d;
    const dim3 block_dim = {
        cuda_calc_num_blocks(size, block_size.x)
    };

    sigmoid_kernel<<<block_dim, block_size>>>(in, out, size);
    CUDA_CHECK_LAST_ERROR();
}

void tanh_gpu(const float* in, float* out, size_t size)
{
    const cuda_props_t* props = get_cuda_props();
    const dim3 block_size = props->default_block_size_1d;
    const dim3 block_dim = {
        cuda_calc_num_blocks(size, block_size.x)
    };

    tanh_kernel<<<block_dim, block_size>>>(in, out, size);
    CUDA_CHECK_LAST_ERROR();
}

void relu_gpu(const float* in, float* out, size_t size)
{
    const cuda_props_t* props = get_cuda_props();
    const dim3 block_size = props->default_block_size_1d;
    const dim3 block_dim = {
        cuda_calc_num_blocks(size, block_size.x)
    };

    relu_kernel<<<block_dim, block_size>>>(in, out, size);
    CUDA_CHECK_LAST_ERROR();
}

void dsigmoid_gpu(const float* in, float* out, size_t size)
{
    const cuda_props_t* props = get_cuda_props();
    const dim3 block_size = props->default_block_size_1d;
    const dim3 block_dim = {
        cuda_calc_num_blocks(size, block_size.x)
    };

    dsigmoid_kernel<<<block_dim, block_size>>>(in, out, size);
    CUDA_CHECK_LAST_ERROR();
}

void dtanh_gpu(const float* in, float* out, size_t size)
{
    const cuda_props_t* props = get_cuda_props();
    const dim3 block_size = props->default_block_size_1d;
    const dim3 block_dim = {
        cuda_calc_num_blocks(size, block_size.x)
    };

    dtanh_kernel<<<block_dim, block_size>>>(in, out, size);
    CUDA_CHECK_LAST_ERROR();
}

void drelu_gpu(const float* in, float* out, size_t size)
{
    const cuda_props_t* props = get_cuda_props();
    const dim3 block_size = props->default_block_size_1d;
    const dim3 block_dim = {
        cuda_calc_num_blocks(size, block_size.x)
    };

    drelu_kernel<<<block_dim, block_size>>>(in, out, size);
    CUDA_CHECK_LAST_ERROR();
}
