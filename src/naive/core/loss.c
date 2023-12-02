
#include "loss.h"

#include <math.h>
#include <stdlib.h>


static void softmaxv(const float* in, float* out, size_t size);


uint32_t LossInit(Loss* loss, const tensor_shape_t* input_shape, size_t max_batch_size, LossFunctionEnum loss_function)
{
    tensor_shape_t max_input_shape = *input_shape;
    max_input_shape.dims[TENSOR_BATCH_DIM] = max_batch_size;
    tensor_allocate(&loss->gradient_mem, &max_input_shape);
    tensor_allocate(&loss->scratch_mem, &max_input_shape);

    switch (loss_function) {
        case LOSS_FUNCTION_MSE:
            loss->function = LossMSE;
            loss->derivative = DLossMSE;
            break;
        case LOSS_FUNCTION_CROSS_ENTROPY:
            loss->function = LossCrossEntropy;
            loss->derivative = DLossCrossEntropy;
            break;
    }

    return 0;
}


uint32_t LossAccuracy(Loss* loss, const tensor_t* input, const uint8_t* labels)
{
    const tensor_shape_t* shape = tensor_get_shape(input);
    const float* input_data = tensor_get_data_const(input);

    const size_t batch_size = shape->dims[TENSOR_BATCH_DIM];
    const size_t channels = shape->dims[TENSOR_CHANNEL_DIM];

    uint32_t accuracy = 0;
    for (size_t i = 0; i < batch_size; i++) {
        const float* channels_input = input_data + i * channels;
        uint32_t prediction = Max(channels_input, channels);
        accuracy += (uint32_t)(prediction == labels[i]);
    }
    return accuracy;
}


float LossCompute(Loss* loss, const tensor_t* input, const uint8_t* labels)
{
    const tensor_shape_t* shape = tensor_get_shape(input);
    const float* input_data = tensor_get_data_const(input);
    float* scratch_data = tensor_get_data(&loss->scratch_mem);

    const size_t batch_size = shape->dims[TENSOR_BATCH_DIM];
    const size_t channels = shape->dims[TENSOR_CHANNEL_DIM];


    float sum = 0.0f;
    for (size_t i = 0; i < batch_size; i++) {
        const float* channels_input = input_data + i * channels;
        float* channels_scratch = scratch_data + i * channels;
        sum += loss->function(channels_input, channels_scratch, channels, labels[i]);
    }
    return sum;
}


void LossBackward(Loss* loss, const tensor_t* input, const uint8_t* labels, tensor_t** out_gradient)
{
    const tensor_shape_t* shape = tensor_get_shape(input);

    /* Construct the output gradient to be same size as input and use scratch gradient_mem as
        buffer. */
    tensor_from_memory(&loss->gradient, shape, tensor_get_data(&loss->gradient_mem));

    const float* input_data = tensor_get_data_const(input);
    float* gradient = tensor_get_data(&loss->gradient);
    float* scratch_data = tensor_get_data(&loss->scratch_mem);

    const size_t batch_size = shape->dims[TENSOR_BATCH_DIM];
    const size_t channels = shape->dims[TENSOR_CHANNEL_DIM];

    for (size_t i = 0; i < batch_size; i++) {
        const float* channels_input = input_data + i * channels;
        float* channels_gradient = gradient + i * channels;
        float* channels_scratch = scratch_data + i * channels;
        loss->derivative(channels_input, channels_scratch, channels_gradient, channels, labels[i]);
    }

    *out_gradient = &loss->gradient;
}


void LossDeinit(Loss* loss)
{
    tensor_destory(&loss->gradient_mem);
}







float LossMSE(const float* v, float* scratch, size_t size, uint32_t label)
{
    float sum = 0.0f;

    for (size_t i = 0; i < size; i++) {
        float t = (label == i) - v[i];
        sum += t * t;
    }
    return sum / size;
}


void DLossMSE(const float* input, float* scratch, float* gradient, size_t size, uint32_t label)
{
    for (size_t i = 0; i < size; i++) {
        gradient[i] = input[i] - (label == i);
    }
}

#define max(x, y) (((x) > (y)) ? (x) : (y))

float LossCrossEntropy(const float* v, float* scratch, size_t size, uint32_t label)
{
    softmaxv(v, scratch, size);
    return -logf(max(1e-12, scratch[label]));
}


void DLossCrossEntropy(const float* input, float* scratch, float* gradient, size_t size, uint32_t label)
{
    for (size_t i = 0; i < size; i++)
        gradient[i] = scratch[i] - (label == i);
}



uint32_t Max(const float* v, size_t size)
{
    uint32_t max = 0;
    for (size_t i = 1; i < size; i++)
        if (v[i] > v[max])
            max = i;
    return max;
}


static void softmaxv(const float* in, float* out, size_t size)
{
    // Find the max value
    float max = -3.4028235e38;
    for (size_t i = 0; i < size; i++)
        if (in[i] > max)
            max = in[i];
    // exp(v - max) all values
    for (size_t i = 0; i < size; i++)
        out[i] = exp(in[i] - max);
    // sum them
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++)
        sum += out[i];
    // divide every value by the sum
    sum = 1.0f / sum;
    for (size_t i = 0; i < size; i++)
        out[i] = out[i] * sum;
}
