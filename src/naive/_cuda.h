#pragma once

#include <stddef.h>
#include <stdint.h>


typedef enum {
    cuda_memcpy_host_to_host = 0,
    cuda_memcpy_host_to_device = 1,
    cuda_memcpy_device_to_host = 2,
    cuda_memcpy_device_to_device = 3,
} cuda_memcpy_kind_t;

#ifdef __cplusplus
extern "C" {
#endif

void* cuda_malloc(size_t size);
uint32_t cuda_memcpy(void* to, void* from, size_t count, cuda_memcpy_kind_t kind);
uint32_t cuda_memset(void* data, int val, size_t count);
uint32_t cuda_free(void* ptr);

#ifdef __cplusplus
}
#endif
