
#include "ai_weight_init.h"

#include "ai_random.h"

#include <math.h>

/* Weight initialsation functions for linear layers */

float AI_LinearWeightInitXavier(size_t input_size, size_t output_size)
{
    return AI_RandomNormal(0.0f, sqrtf(1.0f / input_size));
}

float AI_LinearWeightInitHe(size_t input_size, size_t output_size)
{
    return AI_RandomNormal(0.0f, sqrtf(2.0f / input_size));
}

/* Bias initialisation functions for linear layers */

float AI_LinearBiasInitZeros(size_t input_size, size_t output_size)
{
    return 0.0f;
}

/* Weight initialsation functions for convolutional layers */

float AI_ConvWeightInitXavier(size_t filter_width, size_t filter_height, size_t input_channels)
{
    return AI_RandomNormal(0.0f, sqrtf(1.0f / (filter_width * filter_height * input_channels)));
}

float AI_ConvWeightInitHe(size_t filter_width, size_t filter_height, size_t input_channels)
{
    return AI_RandomNormal(0.0f, sqrtf(2.0f / (filter_width * filter_height * input_channels)));
}

/* Bias initialsation functions for convolutional layers */

float AI_ConvBiasInitZeros(size_t filter_width, size_t filter_height, size_t input_channels)
{
    return 0.0f;
}
