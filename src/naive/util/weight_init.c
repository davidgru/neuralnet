
#include "weight_init.h"

#include "random.h"

#include <math.h>

/* Weight initialsation functions for linear layers */

float linear_weight_init_xavier(size_t input_size, size_t output_size)
{
    return RandomNormal(0.0f, sqrtf(1.0f / input_size));
}

float linear_weight_init_he(size_t input_size, size_t output_size)
{
    return RandomNormal(0.0f, sqrtf(2.0f / input_size));
}

/* Bias initialisation functions for linear layers */

float linear_bias_init_zeros(size_t input_size, size_t output_size)
{
    return 0.0f;
}

/* Weight initialsation functions for convolutional layers */

float conv_weight_init_xavier(size_t filter_width, size_t filter_height, size_t input_channels)
{
    return RandomNormal(0.0f, sqrtf(1.0f / (filter_width * filter_height * input_channels)));
}

float conv_weight_init_he(size_t filter_width, size_t filter_height, size_t input_channels)
{
    return RandomNormal(0.0f, sqrtf(2.0f / (filter_width * filter_height * input_channels)));
}

/* Bias initialsation functions for convolutional layers */

float conv_bias_init_zeros(size_t filter_width, size_t filter_height, size_t input_channels)
{
    return 0.0f;
}
