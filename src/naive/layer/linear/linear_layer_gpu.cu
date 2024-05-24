#include "layer/linear/linear_layer_internal.h"

#include "_cuda.h"

__global__
void matmul(const float* m1, const float* m2, float* output, int height1, int width2, int sharedDim)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < height1 && c < width2) {
        float sum = 0.0f;
        for (int s = 0; s < sharedDim; s++) {
            sum += m1[r * sharedDim + s] * m2[s * width2 + c];
        }
        output[r * width2 + c] = sum;
    }
}


__global__
void matmul_t1(const float* m1, const float* m2, float* output, int width1, int width2, int sharedDim)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < width1 && c < width2) {
        float sum = 0.0f;
        for (int s = 0; s < sharedDim; s++) {
            sum += m1[s * width1 + r] * m2[s * width2 + c];
        }
        output[r * width2 + c] = sum;
    }
}


__global__
void matmul_t2(const float* m1, const float* m2, float* output, int height1, int height2, int sharedDim)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < height1 && c < height2) {
        float sum = 0.0f;
        for (int s = 0; s < sharedDim; s++) {
            sum += m1[r * sharedDim + s] * m2[c * sharedDim + s];
        }
        output[r * height2 + c] = sum;
    }
}


void matrix_product_gpu(const float* m1, const float* m2, float* output, size_t height1, size_t width2, size_t sharedDim)
{
    const cuda_props_t* props = get_cuda_props();

    const dim3 block_size = props->default_block_size_2d;
    const dim3 num_blocks = {
        cuda_calc_num_blocks((unsigned int)height1, block_size.x),
        cuda_calc_num_blocks((unsigned int)width2, block_size.y),
        1
    };

    matmul<<<num_blocks, block_size>>>(m1, m2, output, height1, width2, sharedDim);
    CUDA_CHECK_LAST_ERROR();
}


void matrix_product_t1_gpu(const float* m1, const float* m2, float* output, size_t width1, size_t width2, size_t sharedDim)
{
    const cuda_props_t* props = get_cuda_props();

    const dim3 block_size = props->default_block_size_2d;
    const dim3 num_blocks = {
        cuda_calc_num_blocks((unsigned int)width1, block_size.x),
        cuda_calc_num_blocks((unsigned int)width2, block_size.y),
        1
    };

    matmul_t1<<<num_blocks, block_size>>>(m1, m2, output, width1, width2, sharedDim);
    CUDA_CHECK_LAST_ERROR();
}


void matrix_product_t2_gpu(const float* m1, const float* m2, float* output, size_t height1, size_t height2, size_t sharedDim)
{
    const cuda_props_t* props = get_cuda_props();

    const dim3 block_size = props->default_block_size_2d;
    const dim3 num_blocks = {
        cuda_calc_num_blocks((unsigned int)height1, block_size.x),
        cuda_calc_num_blocks((unsigned int)height2, block_size.y),
        1
    };

    matmul_t2<<<num_blocks, block_size>>>(m1, m2, output, height1, height2, sharedDim);
    CUDA_CHECK_LAST_ERROR();
}
