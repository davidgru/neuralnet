
#include "ai_loss.h"

#include <math.h>
#include <stdlib.h>


uint32_t AI_LossInit(AI_Loss* loss, size_t size, size_t mini_batch_size, AI_LossFunctionEnum loss_function)
{
    loss->size = size;
    loss->mini_batch_size = mini_batch_size;
    
    loss->gradient = (float*)malloc(size * mini_batch_size * sizeof(float));
    if (!loss->gradient)
        return 1;

    switch (loss_function) {
        case AI_LOSS_FUNCTION_MSE:
            loss->function = AI_LossMSE;
            loss->derivative = AI_DLossMSE;
            break;
        case AI_LOSS_FUNCTION_CROSS_ENTROPY:
            loss->function = AI_LossCrossEntropy;
            loss->derivative = AI_DLossCrossEntropy;
            break;
    }
    return 0;
}

void AI_LossLink(AI_Loss* loss, AI_Layer* layer)
{
    loss->input = layer->output;
    layer->prev_gradient = loss->gradient;
}

uint32_t AI_LossAccuracy(AI_Loss* loss, uint8_t* labels)
{
    uint32_t accuracy = 0;
    for (size_t i = 0; i < loss->mini_batch_size; i++)
        accuracy = accuracy + (AI_Max(loss->input + i * loss->size, loss->size) == labels[i]);
    return accuracy;
}


float AI_LossCompute(AI_Loss* loss, uint8_t* labels)
{
    float sum = 0.0f;
    for (size_t i = 0; i < loss->mini_batch_size; i++)
        sum += loss->function(loss->input + i * loss->size, loss->size, labels[i]);
    return sum;
}


void AI_LossBackward(AI_Loss* loss, uint8_t* labels)
{
    for (size_t i = 0; i < loss->mini_batch_size; i++)
        loss->derivative(loss->input + i * loss->size, loss->gradient + i * loss->size, loss->size, labels[i]);
}


void AI_LossDeinit(AI_Loss* loss)
{
    if (loss->gradient)
        free(loss->gradient);
}







float AI_LossMSE(float* v, size_t size, uint32_t label)
{
    float sum = 0.0f;

    for (size_t i = 0; i < size; i++) {
        float t = (label == i) - v[i];
        sum += t * t;
    }
    return sum / size;
}


void AI_DLossMSE(float* input, float* gradient, size_t size, uint32_t label)
{
    for (size_t i = 0; i < size; i++) {
        gradient[i] = input[i] - (label == i);
    }
}

#define max(x, y) (((x) > (y)) ? (x) : (y))

float AI_LossCrossEntropy(float* v, size_t size, uint32_t label)
{
    return -logf(max(1e-12, v[label]));
}


void AI_DLossCrossEntropy(float* input, float* gradient, size_t size, uint32_t label)
{
    for (size_t i = 0; i < size; i++)
        gradient[i] = input[i] - (label == i);
}



uint32_t AI_Max(float* v, size_t size)
{
    uint32_t max = 0;
    for (size_t i = 1; i < size; i++)
        if (v[i] > v[max])
            max = i;
    return max;
}

