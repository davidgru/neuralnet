#include <stdlib.h>
#include <string.h>

#include "color_augment.h"

#include "log.h"
#include "random.h"

#include "util/ai_math.h"


typedef struct {
    color_augment_config_t config;
} color_aug_context_t;


static void color_aug_init(augment_context_t* context, const augment_config_t* config)
{
    color_aug_context_t* aug_context = context;

    aug_context->config = *(const color_augment_config_t*)config;
}


static void color_aug_inplace(augment_context_t* context, tensor_t* input_output)
{
    color_aug_context_t* aug_context = context;


    const tensor_shape_t* shape = tensor_get_shape(input_output);

    const int32_t batch_size = tensor_shape_get_dim(shape, TENSOR_BATCH_DIM);
    const int32_t channels = tensor_shape_get_dim(shape, TENSOR_CHANNEL_DIM);
    const int32_t height = tensor_shape_get_dim(shape, TENSOR_HEIGHT_DIM);
    const int32_t width = tensor_shape_get_dim(shape, TENSOR_WIDTH_DIM);

    const int32_t channel_size = height * width;
    const int32_t per_batch_size = channels * channel_size;


    float* data = tensor_get_data(input_output);
    for (int32_t n = 0; n < batch_size; n++) {
        float* current_image = &data[n * per_batch_size];

        /* Adjusting brightness: img = factor * img, factor ~ N(1, brightness_std) */
        if (RandomUniform(0.0f, 1.0f) < aug_context->config.brightness_augment_prob) {
            const float factor = RandomNormal(0.0f, aug_context->config.brightness_std);
            VectorAddScalar(current_image, factor, per_batch_size);
        }

        /* Adjust contrast: img = mid + factor * (img - mid), factor ~ N(1, contrast_std) */
        if (RandomUniform(0.0f, 1.0f) < aug_context->config.contrast_augment_prob) {
            const float factor = RandomNormal(1.0f, aug_context->config.contrast_std);
            VectorAddScalar(current_image, -aug_context->config.contrast_midpoint, per_batch_size);
            VectorScale(current_image, factor, per_batch_size);
            VectorAddScalar(current_image, aug_context->config.contrast_midpoint, per_batch_size);
        }
    }

}


const augment_impl_t aug_color = {
    .init_func = color_aug_init,
    .deinit_func = NULL,
    .augment_func = NULL,
    .augment_inplace_func = color_aug_inplace,
    .context_size = sizeof(color_aug_context_t),
};
