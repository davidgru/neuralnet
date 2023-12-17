#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* Definition of weight and bias initialisation functions for linear layers */

typedef float (*linear_weight_init_func_t)(size_t input_size, size_t output_size);
typedef float (*linear_bias_init_func_t)(size_t input_size, size_t output_size);

/* Definition of weight and bias initialisation functions for convolutional layers*/

typedef float (*conv_weight_init_func_t)(size_t input_width, size_t input_height, size_t input_channels);
typedef float (*conv_bias_init_func_t)(size_t input_width, size_t input_height, size_t input_channels);


/* Weight initialsation functions for linear layers */

float linear_weight_init_xavier(size_t input_size, size_t output_size);
float linear_weight_init_he(size_t input_size, size_t output_size);

/* Bias initialisation functions for linear layers */

float linear_bias_init_zeros(size_t input_size, size_t output_size);

/* Weight initialsation functions for convolutional layers */

float conv_weight_init_xavier(size_t filter_width, size_t filter_height, size_t input_channels);
float conv_weight_init_he(size_t filter_width, size_t filter_height, size_t input_channels);

/* Bias initialsation functions for convolutional layers */

float conv_bias_init_zeros(size_t filter_width, size_t filter_height, size_t input_channels);
