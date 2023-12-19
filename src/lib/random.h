#pragma once


#include "tensor.h"


float RandomNormal(float mean, float stddev);

float RandomUniform(float min, float max);


void random_mask(tensor_t* tensor, float ratio);
