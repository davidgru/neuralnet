#include "_cuda.h"

extern "C" {
#include "log.h"
}

static cudaMemcpyKind map_memcpy_kind(cuda_memcpy_kind_t kind) {
    switch (kind) {
        case cuda_memcpy_host_to_host: return cudaMemcpyHostToHost;
        case cuda_memcpy_host_to_device: return cudaMemcpyHostToDevice;
        case cuda_memcpy_device_to_host: return cudaMemcpyDeviceToHost;
        case cuda_memcpy_device_to_device: return cudaMemcpyDeviceToDevice;
        default: return cudaMemcpyDefault;
    }
}


static bool cuda_props_init;
static cuda_props_t cuda_props;


const cuda_props_t* get_cuda_props()
{
    if (cuda_props_init) {
        return &cuda_props;
    }

    /* TODO check cuda device props */

    cuda_props.default_block_size_1d.x = 256;
    cuda_props.default_block_size_1d.y = 1;
    cuda_props.default_block_size_1d.z = 1;

    cuda_props.default_block_size_2d.x = 16;
    cuda_props.default_block_size_2d.y = 16;
    cuda_props.default_block_size_2d.z = 1;

    return &cuda_props;
}


uint32_t cuda_check_error(cudaError_t error, const char* file, int line, bool abort)
{
    if (error != cudaSuccess) {
        LOG_ERROR("%s:%d cuda error: %s\n", file, line, cudaGetErrorString(error));
        if (abort) exit(error);
        return 1;
    }
    return 0;
}


void* cuda_malloc(size_t size)
{
    void* dev_ptr = NULL;

    cudaError_t err = cudaMalloc(&dev_ptr, size);
    CUDA_CHECK_ERROR(err);
    if (err != cudaSuccess) {
        dev_ptr = NULL;
    }

    return dev_ptr;
}


uint32_t cuda_memcpy(void* to, const void* from, size_t count, cuda_memcpy_kind_t kind)
{
    CUDA_CHECK_ERROR(cudaMemcpy(to, from, count, map_memcpy_kind(kind)));
    return 0;
}


uint32_t cuda_memset(void* data, int val, size_t count)
{
    CUDA_CHECK_ERROR(cudaMemset(data, val, count));
    return 0;
}


uint32_t cuda_free(void* ptr)
{
    CUDA_CHECK_ERROR(cudaFree(ptr));
    return 0;
}
