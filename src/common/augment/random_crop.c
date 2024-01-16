#include <stdlib.h>
#include <string.h>

#include "random_crop.h"

#include "log.h"
#include "random.h"


typedef struct {
    random_crop_config_t config;
    float* scratch;
} random_crop_context_t;


static void random_crop_init(augment_context_t* context, const augment_config_t* config)
{
    random_crop_context_t* crop_context = context;

    crop_context->config = *(const random_crop_config_t*)config;
    crop_context->scratch = NULL;
}


static void random_crop_deinit(augment_context_t* context)
{
    random_crop_context_t* crop_context = context;

    if (crop_context->scratch != NULL) {
        free(crop_context->scratch);
    }
}


void crop_image(
    const float* input,
    float* output,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t startY,
    int32_t startX
)
{
    const int32_t channel_size = height * width;

    for (int32_t ch = 0; ch < channels; ch++) {
        for (int32_t r = 0; r < height; r++) {
            for (int32_t c = 0; c < width; c++) {
                const int32_t input_r = r + startY;
                const int32_t input_c = c + startX;

                float out = 0.0f;
                if (input_r >= 0 && input_r < height && input_c >= 0 && input_c < width) {
                    out = input[ch * channel_size + input_r * width + input_c];
                }
                output[ch * channel_size + r * width + c] = out;
            }
        }
    }

}


static void random_crop_inplace(augment_context_t* context, tensor_t* input_output)
{
    random_crop_context_t* crop_context = context;


    const tensor_shape_t* shape = tensor_get_shape(input_output);

    const int32_t batch_size = tensor_shape_get_dim(shape, TENSOR_BATCH_DIM);
    const int32_t channels = tensor_shape_get_dim(shape, TENSOR_CHANNEL_DIM);
    const int32_t height = tensor_shape_get_dim(shape, TENSOR_HEIGHT_DIM);
    const int32_t width = tensor_shape_get_dim(shape, TENSOR_WIDTH_DIM);

    const int32_t channel_size = height * width;
    const int32_t per_batch_size = channels * channel_size;


    if (crop_context->scratch == NULL) {
        crop_context->scratch = calloc(per_batch_size, sizeof(float));
        if (crop_context->scratch == NULL) {
            LOG_ERROR("Mem alloc failed\n");
            return;
        }
    }


    float* data = tensor_get_data(input_output);
    for (int32_t n = 0; n < batch_size; n++) {

        float* current_input = &data[n * per_batch_size];

        /* indices in the unpadded input */
        

        int32_t start_y = RandomUniform(0.0f, 1.0f) * (2.0f * crop_context->config.padding + 1.0f)
            - crop_context->config.padding;
        int32_t start_x = RandomUniform(0.0f, 1.0f) * (2.0f * crop_context->config.padding + 1.0f)
            - crop_context->config.padding;

        crop_image(current_input, crop_context->scratch, channels, height, width, start_y, start_x);

        /* copy cropped image back to input_output */
        memcpy(current_input, crop_context->scratch, per_batch_size * sizeof(float));

    }
}


const augment_impl_t aug_random_crop = {
    .init_func = random_crop_init,
    .deinit_func = random_crop_deinit,
    .augment_func = NULL,
    .augment_inplace_func = random_crop_inplace,
    .context_size = sizeof(random_crop_context_t),
};
