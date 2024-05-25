#include "tensor/tensor_impl.h"
#include "pooling_layer_internal.h"

#include <math.h>

static void pooling_operation_average(const float* x, float* y, size_t input_width, size_t input_height, size_t kernel_width)
{
    const size_t output_width = input_width / kernel_width;
    const size_t output_height = input_height / kernel_width;

    for (size_t i = 0; i < output_height; i++) {
        for (size_t j = 0; j < output_width; j++) {
            float v = 0.0f;
            for (size_t ii = 0; ii < kernel_width; ii++) {
                for (size_t jj = 0; jj < kernel_width; jj++) {
                    v += x[(kernel_width * i + ii) * input_width + (kernel_width * j + jj)];
                }
            }
            y[i * output_width + j] += v / (kernel_width * kernel_width);
        }
    }
}


static void pooling_operation_max(const float* x, float* y, size_t input_width, size_t input_height, size_t kernel_width)
{
    const size_t output_width = input_width / kernel_width;
    const size_t output_height = input_height / kernel_width;

    for (size_t i = 0; i < output_height; i++) {
        for (size_t j = 0; j < output_width; j++) {
            float v = -1e12;
            for (size_t ii = 0; ii < kernel_width; ii++) {
                for (size_t jj = 0; jj < kernel_width; jj++) {
                    v = fmax(v, x[(kernel_width * i + ii) * input_width + (kernel_width * j + jj)]);
                }
            }
            y[i * output_width + j] += v;
        }
    }
}


static void pooling_operation_min(const float* x, float* y, size_t input_width, size_t input_height, size_t kernel_width)
{
    const size_t output_width = input_width / kernel_width;
    const size_t output_height = input_height / kernel_width;

    for (size_t i = 0; i < output_height; i++) {
        for (size_t j = 0; j < output_width; j++) {
            float v = 1e12;
            for (size_t ii = 0; ii < kernel_width; ii++) {
                for (size_t jj = 0; jj < kernel_width; jj++) {
                    v = fmin(v, x[(kernel_width * i + ii) * input_width + (kernel_width * j + jj)]);
                }
            }
            y[i * output_width + j] += v;
        }
    }
}


static void pooling_operation_average_backward(const float* x, const float* dy, float* dx, size_t input_width, size_t input_height, size_t kernel_width)
{
    const size_t output_width = input_width / kernel_width;
    const size_t output_height = input_height / kernel_width;

    for (size_t i = 0; i < output_height; i++) {
        for (size_t j = 0; j < output_width; j++) {
            for (size_t ii = 0; ii < kernel_width; ii++) {
                for (size_t jj = 0; jj < kernel_width; jj++) {
                    dx[(2 * i + ii) * input_width + 2 * j + jj] += dy[i * output_width + j] / (kernel_width * kernel_width);
                }
            }
        }
    }
}


static void pooling_operation_max_backward(const float* x, const float* dy, float* dx, size_t input_width, size_t input_height, size_t kernel_width)
{
    const size_t output_width = input_width / kernel_width;
    const size_t output_height = input_height / kernel_width;

    for (size_t i = 0; i < output_height; i++) {
        for (size_t j = 0; j < output_width; j++) {
            // Find the maximum value and it's position in a kernel sized block
            uint32_t argmax_i = 0;
            uint32_t argmax_j = 0;
            float max = -1e12;
            for (size_t ii = 0; ii < kernel_width; ii++) {
                for (size_t jj = 0; jj < kernel_width; jj++) {
                    uint32_t _i = (2 * i + ii) * input_width + 2 * j + jj;
                    if (x[_i] > max) {
                        max = x[_i];
                        argmax_i = ii;
                        argmax_j = jj;
                        // Store 0 as gradient everywhere
                        dx[_i] += 0;
                    }
                }
            }
            // Overwrite the gradient at the correct position
            dx[(2 * i + argmax_i) * input_width + 2 * j + argmax_j] += dy[i * output_width + j];
        }
    }
}


static void pooling_operation_min_backward(const float* x, const float* dy, float* dx, size_t input_width, size_t input_height, size_t kernel_width)
{
    const size_t output_width = input_width / kernel_width;
    const size_t output_height = input_height / kernel_width;

    for (size_t i = 0; i < output_height; i++) {
        for (size_t j = 0; j < output_width; j++) {
            // Find the minimum value and it's position in a kernel sized block
            uint32_t argmax_i = 0;
            uint32_t argmax_j = 0;
            float max = 1e12;
            for (size_t ii = 0; ii < kernel_width; ii++) {
                for (size_t jj = 0; jj < kernel_width; jj++) {
                    uint32_t _i = (2 * i + ii) * input_width + 2 * j + jj;
                    if (x[_i] < max) {
                        max = x[_i];
                        argmax_i = ii;
                        argmax_j = jj;
                    }
                    // Store 0 as gradient everywhere
                    dx[_i] += 0;
                }
            }
            // Overwrite the gradient at the correct position
            dx[(2 * i + argmax_i) * input_width + 2 * j + argmax_j] += dy[i * output_width + j];
        }
    }

}


void pooling_forward_cpu(const tensor_t* input, tensor_t* output, size_t kernel_width, pooling_kind_t kind)
{
    const float* x = input->data;
    float* y = output->data;

    for (size_t i = 0; i < tensor_batch_size(input); i++) {
        for (size_t j = 0; j < tensor_channels(input); j++) {
            const float* _x = x + i * tensor_per_batch_size(input) + j * tensor_per_channel_size(input);
            float* _y = y + i * tensor_per_batch_size(output) + j * tensor_per_channel_size(output);
            switch(kind) {
                case POOLING_AVERAGE:
                    pooling_operation_average(_x, _y, tensor_width(input), tensor_height(input), kernel_width);
                    break;
                case POOLING_MAX:
                    pooling_operation_max(_x, _y, tensor_width(input), tensor_height(input), kernel_width);
                    break;
                case POOLING_MIN:
                    pooling_operation_min(_x, _y, tensor_width(input), tensor_height(input), kernel_width);
                    break;
            }
        }
    }
}


void pooling_backward_cpu(const tensor_t* input, const tensor_t* prev_grad, tensor_t* grad, size_t kernel_width, pooling_kind_t kind)
{
    const float* x = input->data;
    const float* dy = prev_grad->data;
    float* dx = grad->data;

    for (size_t i = 0; i < tensor_batch_size(input); i++) {
        for (size_t j = 0; j < tensor_channels(input); j++) {
            const float* _x = x + i * tensor_per_batch_size(input) + j * tensor_per_channel_size(input);
            const float* _dy = dy + i * tensor_per_batch_size(prev_grad) + j * tensor_per_channel_size(prev_grad);
            float* _dx = dx + i * tensor_per_batch_size(grad) + j * tensor_per_channel_size(grad);
            switch(kind) {
                case POOLING_AVERAGE:
                    pooling_operation_average_backward(_x, _dy, _dx, tensor_width(input), tensor_height(input), kernel_width);
                    break;
                case POOLING_MAX:
                    pooling_operation_max_backward(_x, _dy, _dx, tensor_width(input), tensor_height(input), kernel_width);
                    break;
                case POOLING_MIN:
                    pooling_operation_min_backward(_x, _dy, _dx, tensor_width(input), tensor_height(input), kernel_width);
                    break;
            }
        }
    }
}
