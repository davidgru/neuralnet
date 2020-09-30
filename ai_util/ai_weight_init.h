#pragma once

#include <stdint.h>

/* Definition of weight and bias initialisation functions for linear layers */

typedef float (*AI_FCLayerWeightInit)(size_t input_size, size_t output_size);
typedef float (*AI_FCLayerBiasInit)(size_t input_size, size_t output_size);

/* Definition of weight and bias initialisation functions for convolutional layers*/

typedef float (*AI_ConvLayerWeightInit)(size_t input_width, size_t input_height, size_t input_channels);
typedef float (*AI_ConvLayerBiasInit)(size_t input_width, size_t input_height, size_t input_channels);


/* Weight initialsation functions for linear layers */

float AI_LinearWeightInitXavier(size_t input_size, size_t output_size);
float AI_LinearWeightInitHe(size_t input_size, size_t output_size);

/* Bias initialisation functions for linear layers */

float AI_LinearBiasInitZeros(size_t input_size, size_t output_size);

/* Weight initialsation functions for convolutional layers */

float AI_ConvWeightInitXavier(size_t filter_width, size_t filter_height, size_t input_channels);
float AI_ConvWeightInitHe(size_t filter_width, size_t filter_height, size_t input_channels);

/* Bias initialsation functions for convolutional layers */

float AI_ConvBiasInitZeros(size_t filter_width, size_t filter_height, size_t input_channels);
