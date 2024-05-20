#include "_cuda.h"


static cudaMemcpyKind map_memcpy_kind(cuda_memcpy_kind_t kind) {
    switch (kind) {
        case cuda_memcpy_host_to_host: return cudaMemcpyHostToHost;
        case cuda_memcpy_host_to_device: return cudaMemcpyHostToDevice;
        case cuda_memcpy_device_to_host: return cudaMemcpyDeviceToHost;
        case cuda_memcpy_device_to_device: return cudaMemcpyDeviceToDevice;
        default: return cudaMemcpyDefault;
    }
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
