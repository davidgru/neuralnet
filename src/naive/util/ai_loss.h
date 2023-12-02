#pragma once


#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "tensor.h"
#include "tensor_impl.h"



typedef float(*AI_LossFunction)(const float* v, float* scratch_mem , size_t size, uint32_t label);
typedef void(*AI_LossDerivative)(const float* in, float* scratch_mem, float* out, size_t size, uint32_t label);


typedef struct AI_Loss {
    AI_LossFunction function;
    AI_LossDerivative derivative;
    tensor_t gradient_mem;
    tensor_t gradient;
    tensor_t scratch_mem;
} AI_Loss;


typedef enum AI_LossFunctionEnum {
    AI_LOSS_FUNCTION_SIGMOID,
    AI_LOSS_FUNCTION_MSE,
    AI_LOSS_FUNCTION_CROSS_ENTROPY,
} AI_LossFunctionEnum;


uint32_t AI_LossInit(AI_Loss* loss, const tensor_shape_t* input_shape, size_t max_batch_size, AI_LossFunctionEnum loss_function);

uint32_t AI_LossAccuracy(AI_Loss* loss, const tensor_t* input, const uint8_t* labels);
float AI_LossCompute(AI_Loss* loss, const tensor_t* input, const uint8_t* labels);

void AI_LossBackward(AI_Loss* loss, const tensor_t* input, const uint8_t* labels, tensor_t** out_gradient);

void AI_LossDeinit(AI_Loss* loss);





float AI_LossMSE(const float* v, float* scratch, size_t size, uint32_t label);
void AI_DLossMSE(const float* input, float* scratch, float* gradient, size_t size, uint32_t label);


float AI_LossCrossEntropy(const float* v, float* scratch, size_t size, uint32_t label);
void AI_DLossCrossEntropy(const float* input, float* scratch, float* gradient, size_t size, uint32_t label);

uint32_t AI_Max(const float* v, size_t size);

