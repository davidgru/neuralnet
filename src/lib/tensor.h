#pragma once


#include <stddef.h>
#include <stdint.h>


/* Maximum amount of dimensions a tensor can have */
#define TENSOR_MAX_DIMS 4

/* Semantic meaning of dimensions in case of input tensor */
#define TENSOR_BATCH_DIM 0
#define TENSOR_CHANNEL_DIM 1
#define TENSOR_HEIGHT_DIM 2
#define TENSOR_WIDTH_DIM 3


/**
 * @brief Specifies tensor shape
 * 
 */
typedef struct tensor_shape {
    /* Dimensions of the tensor. Set unused dimensions to zero. */
    size_t dims[TENSOR_MAX_DIMS];
} tensor_shape_t;


/**
 * @brief Compute total number of elements from a shape
 * 
 * @param shape     A shape
 * @return          size of flattened shape 
 */
size_t tensor_size_from_shape(const tensor_shape_t* shape);



/**
 * @brief A tensor handle
 * 
 */
typedef struct tensor tensor_t;


/**
 * @brief Allocate tensor resources
 * 
 * @param tensor    A tensor
 * @param shape     The tensor shape
 * @return uint32_t 
 */
uint32_t tensor_allocate(tensor_t* tensor, const tensor_shape_t* shape);



/**
 * @brief Create tensor object as a wrapper to existing memory.
 * 
 * @param tensor    A tensor
 * @param shape     The shape of the memory
 * @param mem       Memory consistent with shape
 * @return uint32_t 
 */
uint32_t tensor_from_memory(tensor_t* tensor, const tensor_shape_t* shape, float* mem);


/**
 * @brief Get number of tensor elements.
 * 
 * @param tensor    A tensor
 * @return          Number of tensor elements
 */
const tensor_shape_t* tensor_get_shape(const tensor_t* tensor);


/**
 * @brief Get tensor data.
 * 
 * @param tensor    A tensor
 * @return A pointer to the tensor data
 */
float* tensor_get_data(tensor_t* tensor);


/**
 * @brief Get tensor data - const version.
 * 
 * @param tensor    A tensor
 * @return A pointer to the tensor data
 */
const float* tensor_get_data_const(const tensor_t* tensor);


/**
 * @brief Free tensor resources
 * 
 * @param tensor A tensor
 * @return uint32_t
 */
uint32_t tensor_destory(tensor_t* tensor);
