#pragma once

#include "tensor/tensor.h"


typedef void (*weight_init_func_t)(tensor_t* tensor);

void winit_xavier(tensor_t* tensor);
void winit_he(tensor_t* tensor);
void winit_zeros(tensor_t* tensor);
