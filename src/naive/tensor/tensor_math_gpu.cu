
extern "C" {
#include "tensor_math_internal.h"
}

#include "_cuda.h"

__global__
void tensor_scale_kernel(float* v, float f, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        v[idx] *= f;
    }
}

__global__
void tensor_eltwise_add_kernel(float* v, const float* w, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        v[idx] += w[idx];
    }
}

__global__
void tensor_eltwise_mul_kernel(float* v, const float* w, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        v[idx] *= w[idx];
    }
}

void tensor_scale_gpu(tensor_t* v, float f)
{
    float* data = tensor_get_data(v);
    unsigned int n = tensor_get_size(v);

    const cuda_props_t* props = get_cuda_props();
    const dim3 block_size = props->default_block_size_1d;
    const dim3 block_dim = {
        cuda_calc_num_blocks(n, block_size.x)
    };

    tensor_scale_kernel<<<block_dim, block_size>>>(data, f, n);
    cuda_check_error();
}

void tensor_eltwise_add_gpu(tensor_t* v, const tensor_t* w)
{
    float* v_data = tensor_get_data(v);
    const float* w_data = tensor_get_data_const(w);
    unsigned int n = tensor_get_size(v);

    const cuda_props_t* props = get_cuda_props();
    const dim3 block_size = props->default_block_size_1d;
    const dim3 block_dim = {
        cuda_calc_num_blocks(n, block_size.x)
    };

    tensor_eltwise_add_kernel<<<block_dim, block_size>>>(v_data, w_data, n);
    cuda_check_error();
}

void tensor_eltwise_mul_gpu(tensor_t* v, const tensor_t* w)
{
    float* v_data = tensor_get_data(v);
    const float* w_data = tensor_get_data_const(w);
    unsigned int n = tensor_get_size(v);

    const cuda_props_t* props = get_cuda_props();
    const dim3 block_size = props->default_block_size_1d;
    const dim3 block_dim = {
        cuda_calc_num_blocks(n, block_size.x)
    };

    tensor_eltwise_mul_kernel<<<block_dim, block_size>>>(v_data, w_data, n);
    cuda_check_error();
}
