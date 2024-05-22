#include <math.h>

#include "activation_layer_internal.h"


void sigmoid_cpu(const float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        out[i] = 1.0f / (1.0f + expf(-in[i]));
    }
}

void tanh_cpu(const float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        out[i] = tanhf(in[i]);
    }
}

void relu_cpu(const float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}

void dsigmoid_cpu(const float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        out[i] = (1.0f - in[i]) * in[i];
    }
}

void dtanh_cpu(const float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        out[i] = 1.0f - (in[i] * in[i]);
    }
}

void drelu_cpu(const float* in, float* out, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        out[i] = in[i] > 0.0f ? 1.0f : 0.0f;
    }
}
