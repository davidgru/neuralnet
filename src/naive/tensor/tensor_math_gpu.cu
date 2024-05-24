#include <cooperative_groups.h>

#include "tensor/tensor_math_internal.h"
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
void tensor_add_scalar_kernel(float* v, float f, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        v[idx] += f;
    }
}

__global__
void tensor_eltwise_scaled_add_kernel(float* v, const float* w, float f, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        v[idx] += f * w[idx];
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

__global__
void tensor_sum_kernel(const float* in_data, float* out_data, int size)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = idx < size ? in_data[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_data[blockIdx.x] = sdata[0];
    };
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
    CUDA_CHECK_LAST_ERROR();
}

void tensor_add_scalar_gpu(tensor_t* v, float f)
{
    float* data = tensor_get_data(v);
    unsigned int n = tensor_get_size(v);

    const cuda_props_t* props = get_cuda_props();
    const dim3 block_size = props->default_block_size_1d;
    const dim3 block_dim = {
        cuda_calc_num_blocks(n, block_size.x)
    };

    tensor_add_scalar_kernel<<<block_dim, block_size>>>(data, f, n);
    CUDA_CHECK_LAST_ERROR();
}


void tensor_eltwise_add_gpu(tensor_t* v, const tensor_t* w)
{
    tensor_scaled_add_gpu(v, w, 1.0f);
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
    CUDA_CHECK_LAST_ERROR();
}

void tensor_scaled_add_gpu(tensor_t* v, const tensor_t* w, float f)
{
    float* v_data = tensor_get_data(v);
    const float* w_data = tensor_get_data_const(w);
    unsigned int n = tensor_get_size(v);

    const cuda_props_t* props = get_cuda_props();
    const dim3 block_size = props->default_block_size_1d;
    const dim3 block_dim = {
        cuda_calc_num_blocks(n, block_size.x)
    };

    tensor_eltwise_scaled_add_kernel<<<block_dim, block_size>>>(v_data, w_data, f, n);
    CUDA_CHECK_LAST_ERROR();
}

static void tensor_sum_pass(const float* in_data, float* out_data, size_t n, size_t block_size, size_t num_blocks)
{
    tensor_sum_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(in_data, out_data, n);
}

void tensor_sum_gpu(tensor_t* v, const tensor_t* w)
{
    float* v_data = tensor_get_data(v);
    const float* w_data = tensor_get_data_const(w);
    unsigned int n = tensor_get_size(w);

    const cuda_props_t* props = get_cuda_props();
    unsigned int block_size = props->default_block_size_1d.x;
    unsigned int num_blocks = cuda_calc_num_blocks(n, block_size);

    if (num_blocks == 1) {
        tensor_sum_pass(w_data, v_data, n, block_size, num_blocks);
    } else {
        float* tmp_data = NULL;
        unsigned int tmp_size = num_blocks;
        tmp_data = (float*)cuda_malloc(2 * tmp_size * sizeof(float));
        tensor_sum_pass(w_data, tmp_data, n, block_size, num_blocks);
        n = num_blocks;
        num_blocks = cuda_calc_num_blocks(n, block_size);

        int it = 0;
        while(num_blocks > 1) {
            tensor_sum_pass(&tmp_data[(it & 1) * tmp_size], &tmp_data[(it ^ 1) * tmp_size], n, block_size, num_blocks);    
            n = num_blocks;
            num_blocks = cuda_calc_num_blocks(n, block_size);
            it++;
        }
        tensor_sum_pass(&tmp_data[(it & 1)], v_data, n, block_size, num_blocks);
        cuda_free(tmp_data);
    }
    CUDA_CHECK_LAST_ERROR();
}
