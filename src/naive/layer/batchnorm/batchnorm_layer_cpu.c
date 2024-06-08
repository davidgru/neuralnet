#include <math.h>

#include "batchnorm_layer_internal.h"

#include "util/ai_math.h"


void batchnorm_forward_cpu(const tensor_t* input, const tensor_t* mean, const tensor_t* var,
    const tensor_t* scale, const tensor_t* shift, tensor_t* output, float bn_eps)
{
    /* Apply normalization */
    for (size_t n = 0; n < tensor_batch_size(input); n++) {
        for (size_t ch = 0; ch < tensor_channels(input); ch++) {
            for (size_t i = 0; i < tensor_per_channel_size(input); i++) {
                size_t offset = (n * tensor_channels(input) + ch) * tensor_per_channel_size(input) + i;
                output->data[offset] = (input->data[offset] - mean->data[ch])
                    / sqrtf(var->data[ch] + bn_eps);
            }
        }
    }

    /* Apply scale and shift */
    if (scale != NULL && shift != NULL) {
        for (size_t n = 0; n < tensor_batch_size(input); n++) {
            for (size_t ch = 0; ch < tensor_channels(input); ch++) {
                for (size_t i = 0; i < tensor_per_channel_size(input); i++) {
                    size_t offset = (n * tensor_channels(input) + ch) * tensor_per_channel_size(input) + i;
                    output->data[offset] = scale->data[ch] * output->data[offset] + shift->data[ch];
                }
            }
        }
    }
}

void batchnorm_backward_weights_cpu(const tensor_t* input, const tensor_t* mean, const tensor_t* var,
    const tensor_t* d_output, tensor_t* d_scale, tensor_t* d_shift, float bn_eps)
{
    /* calculate gradient of scale gamma and shift beta */
    for (size_t ch = 0; ch < tensor_channels(input); ch++) {
        float d_scale_acc = 0.0f;
        float d_shift_acc = 0.0f;
        for (size_t n = 0; n < tensor_batch_size(input); n++) {
            for (size_t i = 0; i < tensor_per_channel_size(input); i++) {
                const size_t idx = (n * tensor_channels(input) + ch) * tensor_per_channel_size(input) + i;
                float x_hat = (input->data[idx] - mean->data[ch]) / sqrtf(var->data[ch] + bn_eps);
                d_scale_acc += d_output->data[idx] * x_hat;
                d_shift_acc += d_output->data[idx];
            }
        }
        d_scale->data[ch] = d_scale_acc / tensor_batch_size(input);
        d_shift->data[ch] = d_shift_acc / tensor_batch_size(input);
    }
}

void batchnorm_backward_data_cpu(const tensor_t* input, const tensor_t* mean, const tensor_t* var,
    const tensor_t* scale, const tensor_t* d_output, tensor_t* tmp_d_mean, tensor_t* tmp_d_var,
    tensor_t* d_input, float bn_eps)
{

    /* calculate d_\hat{x} = d_y * gamma. can make use of d_x memory */
    tensor_set_zero(d_input);
    for (size_t n = 0; n < tensor_batch_size(input); n++) {
        for (size_t ch = 0; ch < tensor_channels(input); ch++) {
            const size_t off = (n * tensor_channels(input) + ch) * tensor_per_channel_size(input);
            VectorScaledAdd(&d_input->data[off], &d_output->data[off], scale->data[ch], tensor_per_channel_size(input));
        }
    }

    /* calculate d_var */
    for (size_t ch = 0; ch < tensor_channels(input); ch++) {
        float acc = 0.0f;
        for (size_t n = 0; n < tensor_batch_size(input); n++) {
            for (size_t i = 0; i < tensor_per_channel_size(input); i++) {
                const size_t idx = (n * tensor_channels(input) + ch) * tensor_per_channel_size(input) + i;
                acc += d_input->data[idx] * (input->data[idx] - mean->data[ch]);
            }
        }
        tmp_d_var->data[ch] = acc * -0.5f * powf(var->data[ch] + bn_eps, -1.5f);
    }

    /* calculate d_mean */
    for (size_t ch = 0; ch < tensor_channels(input); ch++) {
        float acc1 = 0.0f; /* sum_{i=1..m} d\hat{x}_i */
        float acc2 = 0.0f; /* sum_{i=1..m} -2.0f * (x_i - mean) */
        for (size_t n = 0; n < tensor_batch_size(input); n++) {
            for (size_t i = 0; i < tensor_per_channel_size(input); i++) {
                const size_t idx = (n * tensor_channels(input) + ch) * tensor_per_channel_size(input) + i;
                acc1 += d_input->data[idx];
                acc2 += -2.0f * (input->data[idx] - mean->data[ch]);
            }
        }
        tmp_d_mean->data[ch] = -acc1 / sqrtf(var->data[ch] + bn_eps)
            + tmp_d_var->data[ch] * acc2 / (tensor_batch_size(input) * tensor_per_channel_size(input));
    }

    /* calculate d_x */
    for (size_t n = 0; n < tensor_batch_size(input); n++) {
        for (size_t ch = 0; ch < tensor_channels(input); ch++) {
            for (size_t i = 0; i < tensor_per_channel_size(input); i++) {
                const size_t idx = (n * tensor_channels(input) + ch) * tensor_per_channel_size(input) + i;
                /* Before this step, d_input stores d_\hat{x} */
                d_input->data[idx] = d_input->data[idx] / sqrtf(var->data[ch] + bn_eps)
                    + tmp_d_var->data[ch] * 2.0f * (input->data[idx] - mean->data[ch])
                        / (tensor_batch_size(input) * tensor_per_channel_size(input))
                    + tmp_d_mean->data[ch] / (tensor_batch_size(input) * tensor_per_channel_size(input));
            }
        }
    }
}
