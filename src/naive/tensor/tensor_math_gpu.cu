#include <curand_kernel.h>

#include "tensor/tensor_math_internal.h"
#include "_cuda.h"


static unsigned int next_pow2(unsigned int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}


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


__global__
void tensor_sum_axis_kernel(const float* in_data, float* out_data, int num_reduce, int reduce_ndim, int outer_stride, int inner_stride)
{
    extern __shared__ float sdata[];
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.z * blockDim.z + threadIdx.z;

    const int xz_dim = blockDim.x * blockDim.z;

    sdata[threadIdx.y * xz_dim + threadIdx.z * blockDim.x + threadIdx.x] = ((j < reduce_ndim) && (k < inner_stride) && (i < num_reduce)) ?
        in_data[i * outer_stride + j * inner_stride + k] : 0.0f;
    __syncthreads();

    for (int s = blockDim.z / 2; s > 0; s >>= 1) {
        if (threadIdx.z < s) {
            sdata[threadIdx.y * xz_dim + threadIdx.z * blockDim.x + threadIdx.x]
                += sdata[threadIdx.y * xz_dim + (threadIdx.z + s) * blockDim.x + threadIdx.x];
        }
        __syncthreads();
    }

    if ((threadIdx.z == 0) && (j < reduce_ndim) && (k < inner_stride) && (i < num_reduce)) {
        out_data[(i * gridDim.z + blockIdx.z) * inner_stride + k] = sdata[threadIdx.y * xz_dim + threadIdx.x];
    };
}


void tensor_sum_axis_gpu(tensor_t* v, const tensor_t* w, size_t outer_stride,
    size_t outer_len, size_t axis_len, size_t inner_stride)
{
    float* v_data = tensor_get_data(v);
    const float* w_data = tensor_get_data_const(w);

    const cuda_props_t* props = get_cuda_props();
    unsigned int total_tpb = props->default_block_size_1d.x;
    unsigned int threads_z = min(64, next_pow2(axis_len));
    unsigned int threads_x = min(total_tpb / threads_z, next_pow2(inner_stride));
    unsigned int threads_y = total_tpb / (threads_x * threads_z);
    dim3 threads = { threads_x, threads_y, threads_z };
    dim3 blocks = {
        cuda_calc_num_blocks(inner_stride, threads.x),
        cuda_calc_num_blocks(outer_len, threads.y),
        cuda_calc_num_blocks(axis_len, threads.z)
    };
    unsigned int shared_size = threads_x * threads_y * threads_z * sizeof(float);
    
    if (blocks.z == 1) {
        tensor_sum_axis_kernel<<<blocks, threads, shared_size>>>(w_data, v_data,
            outer_len, axis_len, outer_stride, inner_stride);
        CUDA_CHECK_LAST_ERROR();
    } else {
        float* tmp_data = NULL;
        size_t tmp_size = blocks.z * outer_len * inner_stride;
        cudaMalloc(&tmp_data, 2 * tmp_size * sizeof(float));
        tensor_sum_axis_kernel<<<blocks, threads, shared_size>>>(w_data, tmp_data,
            outer_len, axis_len, outer_stride, inner_stride);
        CUDA_CHECK_LAST_ERROR();
    
        size_t new_in_dims = blocks.z;
        blocks.z = cuda_calc_num_blocks(new_in_dims, threads.z);
        outer_stride = inner_stride * new_in_dims;

        size_t it = 0;
        while(blocks.z > 1) {
            tensor_sum_axis_kernel<<<blocks, threads, shared_size>>>(
                &tmp_data[(it & 1) * tmp_size], &tmp_data[(it ^ 1) * tmp_size],
                outer_len, new_in_dims, outer_stride, inner_stride);
            CUDA_CHECK_LAST_ERROR();
            new_in_dims = blocks.z;
            blocks.z = cuda_calc_num_blocks(new_in_dims, threads.z);
            outer_stride = inner_stride * new_in_dims;
            it++;
        }

        tensor_sum_axis_kernel<<<blocks, threads, shared_size>>>(
            &tmp_data[(it & 1) * tmp_size], v_data,
            outer_len, new_in_dims, outer_stride, inner_stride);
        CUDA_CHECK_LAST_ERROR();
        cudaFree(tmp_data);
    }
}


static constexpr int curand_num_threads = 1024;
static constexpr int seed = 42;
static bool curand_initialized = false;
__device__ curandState curand_states[curand_num_threads];


__global__
void random_init_kernel(int curand_num_threads, unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < curand_num_threads) {
        curand_init(seed, idx, 0, &curand_states[idx]);
    }
}

__global__
void random_mask_kernel(float* output, int curand_num_threads, int size, float ratio)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < curand_num_threads) {
        for (int i = idx; i < size; i += curand_num_threads) {
            output[i] = (curand_uniform(&curand_states[idx]) < ratio) ? 1.0f : 0.0f;
        }
    }
}


void tensor_random_mask_gpu(tensor_t* v, float ratio)
{
    const cuda_props_t* props = get_cuda_props();
    const unsigned int num_threads = props->default_block_size_1d.x;
    const unsigned int num_blocks = cuda_calc_num_blocks(curand_num_threads, num_threads);

    if (!curand_initialized) {
        random_init_kernel<<<num_blocks, num_threads>>>(curand_num_threads, seed);
        curand_initialized = true;
    }

    const size_t size = tensor_get_size(v);
    float* data = tensor_get_data(v);
    random_mask_kernel<<<num_blocks, num_threads>>>(data, curand_num_threads, size, ratio);
}
