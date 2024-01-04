#pragma once


#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "tensor.h"
#include "tensor_impl.h"



typedef float(*LossFunction)(const float* v, float* scratch_mem , size_t size, uint32_t label);
typedef void(*LossDerivative)(const float* in, float* scratch_mem, float* out, size_t size, uint32_t label);


typedef struct {
    LossFunction function;
    LossDerivative derivative;
    tensor_t gradient_mem;
    tensor_t gradient;
    tensor_t scratch_mem;
} Loss;


typedef enum {
    LOSS_FUNCTION_SIGMOID,
    LOSS_FUNCTION_MSE,
    LOSS_FUNCTION_CROSS_ENTROPY,
} LossFunctionEnum;


uint32_t LossInit(Loss* loss, const tensor_shape_t* input_shape, size_t max_batch_size, LossFunctionEnum loss_function);

uint32_t LossAccuracy(Loss* loss, const tensor_t* input, const uint8_t* labels);
float LossCompute(Loss* loss, const tensor_t* input, const uint8_t* labels);

void LossBackward(Loss* loss, const tensor_t* input, const uint8_t* labels, tensor_t** out_gradient);

void LossDeinit(Loss* loss);

void softmaxv(const float* in, float* out, size_t size);
uint32_t argmax(const float* v, size_t size);






float LossMSE(const float* v, float* scratch, size_t size, uint32_t label);
void DLossMSE(const float* input, float* scratch, float* gradient, size_t size, uint32_t label);


float LossCrossEntropy(const float* v, float* scratch, size_t size, uint32_t label);
void DLossCrossEntropy(const float* input, float* scratch, float* gradient, size_t size, uint32_t label);

