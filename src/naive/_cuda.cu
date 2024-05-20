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


uint32_t cuda_check_error()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR("cuda error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}


void* cuda_malloc(size_t size)
{
    void* dev_ptr = NULL;

    cudaError_t err = cudaMalloc(&dev_ptr, size);
    if (err != cudaSuccess) {
        dev_ptr = NULL;
    }

    return dev_ptr;
}


uint32_t cuda_memcpy(void* to, void* from, size_t count, cuda_memcpy_kind_t kind)
{
    return cudaMemcpy(to, from, count, map_memcpy_kind(kind)) != cudaSuccess;
}


uint32_t cuda_memset(void* data, int val, size_t count)
{
    return cudaMemset(data, val, count) != cudaSuccess;
}


uint32_t cuda_free(void* ptr)
{
    return cudaFree(ptr) != cudaSuccess;
}
