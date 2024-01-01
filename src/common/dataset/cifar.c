#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "log.h"

#include "cifar.h"

#include "tensor_impl.h"



#define TRAIN_BATCH_FN_TEMPLATE "data_batch_%d.bin"
#define NUM_BATCH_FILES 5
#define TEST_BATCH_FN           "test_batch.bin"
#define FILENAME_MAXLEN 1024 /* do not trust FILENAME_MAX */

#define IMAGES_PER_BATCH 10000
#define NUM_TRAIN_IMAGES (NUM_BATCH_FILES * IMAGES_PER_BATCH)
#define NUM_TEST_IMAGES IMAGES_PER_BATCH

#define IMAGE_HEIGHT 32
#define IMAGE_WIDTH 32
#define IMAGE_CHANNELS 3
#define IMAGE_SIZE (IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS)


typedef struct {
    float* data;
    uint8_t* labels;

} cifar_context_t;


/* will write at most FILENAME_MAXLEN bytes to out_joined */
static void path_join(const char* base_path, const char* filename, char* out_joined)
{
    const size_t base_path_len = strlen(base_path);
    const size_t filename_len = strlen(filename);
    const size_t endswith_slash = base_path[base_path_len - 1] == '/';

    const size_t total_filename_len = base_path_len + !endswith_slash + filename_len;
    if (total_filename_len >= FILENAME_MAXLEN) {
        LOG_ERROR("Filename too long\n");
        return;
    }

    memcpy(out_joined, base_path, base_path_len);
    if (!endswith_slash) {
        out_joined[base_path_len] = '/';
    }
    memcpy(out_joined + base_path_len + !endswith_slash, filename, filename_len);
    out_joined[total_filename_len] = '\0';
}


static uint32_t read_batch_file(const char* base_path, const char* filename, float* data, uint8_t* labels)
{
    char joined_filename[FILENAME_MAXLEN];
    path_join(base_path, filename, joined_filename);
    
    FILE* file = NULL;
    uint8_t* tmp_image = NULL;

    file = fopen(joined_filename, "r");
    if (file == NULL) {
        LOG_ERROR("Error opening file %s\n", joined_filename);
        goto error;
    }

    tmp_image = malloc(IMAGE_SIZE); /* possibly would fit on stack */
    if (tmp_image == NULL) {
        LOG_ERROR("Out of memory\n");
        goto error;
    }

    for (size_t i = 0; i < IMAGES_PER_BATCH; i++) {
        /* read the label */
        if (fread(&labels[i], 1, 1, file) != 1) {
            LOG_ERROR("Failed reading label from file\n");
            goto error;
        }

        /* read the image */
        if (fread(tmp_image, 1, IMAGE_SIZE, file) != IMAGE_SIZE) {
            LOG_ERROR("Failed reading image from file %s at %zu\n", filename, i);
            goto error;
        }

        /* convert the image to float */
        for (size_t j = 0; j < IMAGE_SIZE; j++) {
            data[i * IMAGE_SIZE + j] = tmp_image[j];
        }
    }

    fclose(file);
    free(tmp_image);
    return 0;
error:
    if (file != NULL) {
        fclose(file);
    }
    if (tmp_image != NULL) {
        free(tmp_image);
    }
    return 1;
}


static uint32_t cifar_init(
    dataset_context_t* context,
    const dataset_create_info_t* create_info,
    tensor_shape_t* out_data_shape
)
{
    cifar_context_t* cifar_context = context;
    const cifar_create_info_t* cifar_create_info = create_info;

    const size_t num_images = cifar_create_info->dataset_kind == TRAIN_SET ? NUM_TRAIN_IMAGES : NUM_TEST_IMAGES;
    *out_data_shape = make_tensor_shape(4, num_images, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH);

    /* allocating memory buffers */
    const size_t data_numelem = num_images * IMAGE_SIZE;
    cifar_context->data = calloc(data_numelem, sizeof(float));
    if (cifar_context->data == NULL) {
        return 1;
    }

    cifar_context->labels = malloc(num_images);
    if (cifar_context->labels == NULL) {
        free(cifar_context->data);
        return 1;
    }

    uint32_t status = 0;
    if (cifar_create_info->dataset_kind == TRAIN_SET) {
        /* read in the NUM_BATCH_FILES batch files */
        char batch_fn[FILENAME_MAXLEN];
        for (int i = 1; i <= NUM_BATCH_FILES; i++) {
            snprintf(batch_fn, FILENAME_MAXLEN, TRAIN_BATCH_FN_TEMPLATE, i);
            float* current_data = &cifar_context->data[(i-1) * IMAGE_SIZE * IMAGES_PER_BATCH];
            uint8_t* current_labels = &cifar_context->labels[(i-1) * IMAGES_PER_BATCH];
            status = read_batch_file(cifar_create_info->path, batch_fn, current_data, current_labels);
            if (status != 0) {
                break;
            }
        }
    } else {
        /* read the single test file */
        status = read_batch_file(cifar_create_info->path, TEST_BATCH_FN, cifar_context->data,
            cifar_context->labels);
    }

    return status;
}


static uint32_t cifar_get_batch(
    dataset_context_t* context,
    const size_t* indices,
    tensor_t* out_batch,
    uint8_t* out_labels
)
{
    cifar_context_t* cifar_context = (cifar_context_t*)context;


    const size_t batch_size = tensor_shape_get_dim(tensor_get_shape(out_batch), TENSOR_BATCH_DIM);
    float* out_data = tensor_get_data(out_batch);

    for (size_t i = 0; i < batch_size; i++) {
        float* out_i = &out_data[IMAGE_SIZE * i];
        float* sample_i = &cifar_context->data[IMAGE_SIZE * indices[i]];
        memcpy(out_i, sample_i, IMAGE_SIZE * sizeof(float));
    
        out_labels[i] = cifar_context->labels[indices[i]];
    }
}


static uint32_t cifar_deinit(dataset_context_t* context)
{

}


const dataset_impl_t cifar_dataset = {
    .init_func = cifar_init,
    .get_batch_func = cifar_get_batch,
    .deinit_func = cifar_deinit,
    .conext_size = sizeof(cifar_context_t)
};
