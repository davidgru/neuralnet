#include "image_flip.h"

#include "log.h"
#include "random.h"


typedef struct {
    float horizontal_flip_prob;
    float vertical_flip_prob;
} aug_flip_context_t;


static void aug_flip_init(augment_context_t* context, const augment_config_t* config);
static void aug_flip(augment_context_t* context, const tensor_t* input, tensor_t* output);
static void aug_flip_inplace(augment_context_t* context, tensor_t* input_output);

static inline void swap(float* a, float* b);


const augment_impl_t aug_image_flip = {
    .init_func = aug_flip_init,
    .deinit_func = NULL,
    .augment_func = NULL,
    .augment_inplace_func = aug_flip_inplace,
    .context_size = sizeof(aug_flip_context_t),
};


static void aug_flip_init(augment_context_t* context, const augment_config_t* config)
{
    aug_flip_context_t* flip_context = context;
    const image_flip_config_t* flip_config = config;

    flip_context->horizontal_flip_prob = flip_config->horizontal_flip_prob;
    flip_context->vertical_flip_prob = flip_config->vertical_flip_prob;
}


static void aug_flip_inplace(augment_context_t* context, tensor_t* input_output)
{
    aug_flip_context_t* flip_context = context;
    

    const tensor_shape_t* shape = tensor_get_shape(input_output);

    const size_t batch_size = tensor_shape_get_dim(shape, TENSOR_BATCH_DIM);
    const size_t channels = tensor_shape_get_dim(shape, TENSOR_CHANNEL_DIM);
    const size_t height = tensor_shape_get_dim(shape, TENSOR_HEIGHT_DIM);
    const size_t width = tensor_shape_get_dim(shape, TENSOR_WIDTH_DIM);


    const size_t channel_size = height * width;
    const size_t per_batch_size = channels * channel_size;

    float* data = tensor_get_data(input_output);


    for (size_t n = 0; n < batch_size; n++) {

        bool flip_horizontal = RandomUniform(0.0f, 1.0f) < flip_context->horizontal_flip_prob;
        bool flip_vertical = RandomUniform(0.0f, 1.0f) < flip_context->vertical_flip_prob;

        for (size_t ch = 0; ch < channels; ch++) {
            float* current_image = data + n * per_batch_size + ch * channel_size;
            
            if (flip_horizontal) {
                for (size_t r = 0; r < height; r++) {
                    for (size_t c = 0; c < width / 2; c++) {
                        swap(&current_image[r * width + c], &current_image[r * width + (width - c - 1)]);
                    }
                }
            }

            if (flip_vertical) {
                for (size_t r = 0; r < height / 2; r++) {
                    for (size_t c = 0; c < width; c++) {
                        swap(&current_image[r * width + c], &current_image[(height - r - 1) * width + c]);
                    }
                }
            }
        }
    }
}


static inline void swap(float* a, float* b)
{
    float tmp = *a;
    *a = *b;
    *b = tmp;
}
