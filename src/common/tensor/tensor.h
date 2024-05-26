#pragma once


#include <stddef.h>
#include <stdint.h>


/* Maximum amount of dimensions a tensor can have */
#define TENSOR_MAX_DIMS 8

/* Semantic meaning of dimensions in case of input tensor */
#define DATA_TENSOR_DIMS 4
#define TENSOR_BATCH_DIM 0
#define TENSOR_CHANNEL_DIM 1
#define TENSOR_HEIGHT_DIM 2
#define TENSOR_WIDTH_DIM 3


/**
 * @brief Specifies tensor shape. tensor_shape implemented by backend
 * 
 */
typedef struct tensor_shape tensor_shape_t;


/**
 * @brief Create a shape object in nchw format
 * 
 * @param ndims     The number of dimensions
 * @param ...       Sizes of dimensions from outer to inner
 * @return tensor_shape_t 
 */
tensor_shape_t make_tensor_shape(size_t ndims, ...);


/**
 * @brief Copy a tensor shape
 * 
 * @param shape     The shape to copy
 * @return tensor_shape_t 
 */
tensor_shape_t copy_tensor_shape(const tensor_shape_t* shape);


/**
 * @brief Destroy a tensor shape
 * 
 * @param shape     The shape to destroy
 */
void destroy_tensor_shape(tensor_shape_t* shape);

/**
 * @brief Get number of shape dimensions.
 * 
 * @param shape     A shape
 * @return size_t 
 */
size_t tensor_shape_get_depth_dim(const tensor_shape_t* shape);


/**
 * @brief Get size of a specific dimension of the shape.
 * 
 * @param shape     A shape
 * @param dim       The dimension (e.g. TENSOR_CHANNEL_DIM)
 * @return size_t 
 */
size_t tensor_shape_get_dim(const tensor_shape_t* shape, size_t dim);


/**
 * @brief Compute total number of elements from a shape
 * 
 * @param shape     A shape
 * @return          size of flattened shape 
 */
size_t tensor_size_from_shape(const tensor_shape_t* shape);


/**
 * @brief Device of a tensor
*/
typedef enum {
    device_cpu = 0,
    device_gpu
} device_t;


/**
 * @brief A tensor handle
 * 
 */
typedef struct tensor tensor_t;


/**
 * @brief Allocate tensor resources on cpu
 * 
 * @param tensor    A tensor
 * @param shape     The tensor shape
 * @return uint32_t 
 */
uint32_t tensor_allocate(tensor_t* tensor, const tensor_shape_t* shape);


/**
 * @brief Allocate tensor resources on device
 * 
 * @param tensor    A tensor
 * @param shape     The tensor shape
 * @param device    Device of the tensor
 * @return uint32_t 
 */
uint32_t tensor_allocate_device(tensor_t* tensor, const tensor_shape_t* shape, device_t device);



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
 * @brief Create tensor object as a wrapper to existing memory.
 * 
 * @param tensor    A tensor
 * @param shape     The shape of the memory
 * @param mem       Memory consistent with shape
 * @param device    Device of the memory
 * @return uint32_t 
 */
uint32_t tensor_from_memory_device(tensor_t* tensor, const tensor_shape_t* shape, float* mem, device_t device);


/**
 * @brief Copy tensor data between two allocated tensors
 * 
 * @param tensor_to     Source tensor of the copy operation
 * @param tensor_from   Target tensor of the copy operation
 * @return uint32_t 
 */
uint32_t tensor_copy(tensor_t* tensor_to, const tensor_t* tensor_from);


/**
 * @brief Set all tensor entries to val.
 * 
 * @param tensor    A tensor
 * @param val       The fill value
*/
uint32_t tensor_fill(tensor_t* tensor, float val);


/**
 * @brief Set the tensor buffer to all zeroes.
 * 
 * @param tensor    A tensor
*/
uint32_t tensor_set_zero(tensor_t* tensor);


/**
 * @brief Get number of tensor elements.
 * 
 * @param tensor    A tensor
 * @return          Number of tensor elements
 */
const tensor_shape_t* tensor_get_shape(const tensor_t* tensor);


/**
 * @brief Get number of tensor elements
 * 
 * @param tensor    A tensor
 * @return          Total number of elements in the tensor
*/
size_t tensor_get_size(const tensor_t* tensor);

/**
 * @brief Get the device of the tensor
 * 
 * @param tensor A tensor
 * @return Device of tensor
*/
device_t tensor_get_device(const tensor_t* tensor);


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
