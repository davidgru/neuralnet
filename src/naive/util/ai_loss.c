
#include "ai_loss.h"

#include <math.h>
#include <stdlib.h>


uint32_t AI_LossInit(AI_Loss* loss, const tensor_shape_t* input_shape, AI_LossFunctionEnum loss_function)
{
    tensor_allocate(&loss->gradient, input_shape);

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
    loss->input = &layer->output;
    layer->prev_gradient = &loss->gradient;
}


uint32_t AI_LossAccuracy(AI_Loss* loss, uint8_t* labels)
{
    const tensor_shape_t* shape = tensor_get_shape(loss->input);
    float* input = tensor_get_data(loss->input);

    const size_t batch_size = shape->dims[TENSOR_BATCH_DIM];
    const size_t channels = shape->dims[TENSOR_CHANNEL_DIM];

    uint32_t accuracy = 0;
    for (size_t i = 0; i < batch_size; i++) {
        float* channels_input = input + i * channels;
        uint32_t prediction = AI_Max(channels_input, channels);
        accuracy += (uint32_t)(prediction == labels[i]);
    }
    return accuracy;
}


float AI_LossCompute(AI_Loss* loss, uint8_t* labels)
{
    const tensor_shape_t* shape = tensor_get_shape(loss->input);
    float* input = tensor_get_data(loss->input);

    const size_t batch_size = shape->dims[TENSOR_BATCH_DIM];
    const size_t channels = shape->dims[TENSOR_CHANNEL_DIM];


    float sum = 0.0f;
    for (size_t i = 0; i < batch_size; i++) {
        float* channels_input = input + i * channels;
        sum += loss->function(channels_input, channels, labels[i]);
    }
    return sum;
}


void AI_LossBackward(AI_Loss* loss, uint8_t* labels)
{
    const tensor_shape_t* shape = tensor_get_shape(loss->input);
    float* input = tensor_get_data(loss->input);
    float* gradient = tensor_get_data(&loss->gradient);

    const size_t batch_size = shape->dims[TENSOR_BATCH_DIM];
    const size_t channels = shape->dims[TENSOR_CHANNEL_DIM];

    for (size_t i = 0; i < batch_size; i++) {
        float* channels_input = input + i * channels;
        float* channels_gradient = gradient + i * channels;
        loss->derivative(channels_input, channels_gradient, channels, labels[i]);
    }
}


void AI_LossDeinit(AI_Loss* loss)
{
    tensor_destory(&loss->gradient);
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

