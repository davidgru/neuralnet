#pragma once

#include <stddef.h>
#include <stdint.h>


#define cuda_calc_num_blocks(total_size, block_size) \
    (unsigned int)(((total_size) + (block_size) - 1) / (block_size))


#ifdef __cplusplus
typedef struct {
    dim3 default_block_size_1d;
    dim3 default_block_size_2d;
} cuda_props_t;

const cuda_props_t* get_cuda_props();
#define CUDA_CHECK_LAST_ERROR() cuda_check_error(cudaGetLastError(), __FILE__, __LINE__)
#define CUDA_CHECK_ERROR(err) cuda_check_error(err, __FILE__, __LINE__)
uint32_t cuda_check_error(cudaError_t error, const char* file, int line, bool abort = true);
#endif


#ifdef __cplusplus
extern "C" {
#endif

void* cuda_malloc(size_t size);

typedef enum {
    cuda_memcpy_host_to_host = 0,
    cuda_memcpy_host_to_device = 1,
    cuda_memcpy_device_to_host = 2,
    cuda_memcpy_device_to_device = 3,
} cuda_memcpy_kind_t;
uint32_t cuda_memcpy(void* to, const void* from, size_t count, cuda_memcpy_kind_t kind);
uint32_t cuda_memset(void* data, int val, size_t count);
uint32_t cuda_free(void* ptr);

#ifdef __cplusplus
}
#endif
