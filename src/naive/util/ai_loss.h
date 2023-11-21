#pragma once


#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "tensor.h"
#include "tensor_impl.h"

#include "layer/ai_base_layer.h"


typedef float(*AI_LossFunction)(float* v, size_t size, uint32_t label);
typedef void(*AI_LossDerivative)(float* in, float* out, size_t size, uint32_t label);


typedef struct AI_Loss {
    AI_LossFunction function;
    AI_LossDerivative derivative;
    tensor_t* input;
    tensor_t gradient;
} AI_Loss;


typedef enum AI_LossFunctionEnum {
    AI_LOSS_FUNCTION_SIGMOID,
    AI_LOSS_FUNCTION_MSE,
    AI_LOSS_FUNCTION_CROSS_ENTROPY,
} AI_LossFunctionEnum;


uint32_t AI_LossInit(AI_Loss* loss, const tensor_shape_t* input_shape, AI_LossFunctionEnum loss_function);
void AI_LossLink(AI_Loss* loss, AI_Layer* layer);

uint32_t AI_LossAccuracy(AI_Loss* loss, uint8_t* labels);
float AI_LossCompute(AI_Loss* loss, uint8_t* labels);

void AI_LossBackward(AI_Loss* loss, uint8_t* labels);

void AI_LossDeinit(AI_Loss* loss);





float AI_LossMSE(float* v, size_t size, uint32_t label);
void AI_DLossMSE(float* input, float* gradient, size_t size, uint32_t label);


float AI_LossCrossEntropy(float* v, size_t size, uint32_t label);
void AI_DLossCrossEntropy(float* input, float* gradient, size_t size, uint32_t label);

uint32_t AI_Max(float* v, size_t size);
