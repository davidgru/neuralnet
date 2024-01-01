#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "mnist.h"
#include "tensor_impl.h"


#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT)

#define IMAGE_MAGIC 2051
#define LABEL_MAGIC 2049


typedef struct {
    float* data;
    uint8_t* labels;
    size_t num_samples;
    size_t sample_size;
} mnist_context_t;


static uint32_t mnist_init(
    dataset_context_t* context,
    const dataset_create_info_t* create_info,
    tensor_shape_t* out_data_shape
);


static uint32_t mnist_get_batch(
    dataset_context_t* context,
    const size_t* indices,
    tensor_t* out_batch,
    uint8_t* out_labels
);


static uint32_t mnist_deinit(dataset_context_t* context);


const dataset_impl_t mnist_dataset = {
    .init_func = mnist_init,
    .get_batch_func = mnist_get_batch,
    .deinit_func = mnist_deinit,
    .conext_size = sizeof(mnist_context_t)
};


typedef struct {
    size_t num_samples;
    size_t image_height;
    size_t image_width;
    const char* image_filename;
    const char* label_filename;
    size_t image_file_size;
    size_t label_file_size;
} mnist_dataset_info_t;


static const mnist_dataset_info_t train_info = {
    .num_samples = 60000,
    .image_height = IMAGE_HEIGHT,
    .image_width = IMAGE_WIDTH,
    .image_filename = "train-images-idx3-ubyte",
    .label_filename = "train-labels-idx1-ubyte",
    .image_file_size = 47040016,
    .label_file_size = 60008
};

static const mnist_dataset_info_t test_info = {
    .num_samples = 10000,
    .image_height = IMAGE_HEIGHT,
    .image_width = IMAGE_WIDTH,
    .image_filename = "t10k-images-idx3-ubyte",
    .label_filename = "t10k-labels-idx1-ubyte",
    .image_file_size = 7840016,
    .label_file_size = 10008
};


static uint32_t flip(uint32_t n);
static FILE* open_file(char* file_path, const char* file_name, size_t file_path_size, size_t file_name_size);
static uint32_t validate_file_size(FILE* fp, size_t required_size);
static uint32_t validate_image_header(FILE* fp, size_t image_sample_size);
static uint32_t validate_label_header(FILE* fp, size_t label_sample_size);
static uint32_t read_images(FILE* fp, float* dest, size_t image_sample_size, size_t padding);
static uint32_t read_image_file(float* dest, char* file_path, const char* file_name, size_t file_path_size, size_t file_name_size, size_t required_file_size, size_t sample_size, size_t padding);
static uint32_t read_label_file(uint8_t* dest, char* file_path, const char* file_name, size_t file_path_size, size_t file_name_size, size_t required_file_size, size_t sample_size);


static uint32_t mnist_init(
    dataset_context_t* context,
    const dataset_create_info_t* create_info,
    tensor_shape_t* out_data_shape
)
{
    mnist_context_t* mnist_context = (mnist_context_t*)context;
    
    const mnist_create_info_t* mnist_create_info = (const mnist_create_info_t*) create_info;
    const mnist_dataset_info_t* dataset_info = mnist_create_info->dataset_kind == TRAIN_SET
        ? &train_info : &test_info;


    const size_t image_height = dataset_info->image_height + 2 * mnist_create_info->padding;
    const size_t image_width = dataset_info->image_width + 2 * mnist_create_info->padding;
    const size_t image_size = image_height * image_width;

    mnist_context->num_samples = dataset_info->num_samples;
    mnist_context->sample_size = image_size;

    *out_data_shape = make_tensor_shape(TENSOR_MAX_DIMS, dataset_info->num_samples, 1, image_height, image_width);

    /* allocate memory for input and labels */
    const size_t total_memory_requirement = tensor_size_from_shape(out_data_shape) * sizeof(float)
        + dataset_info->num_samples * sizeof(uint8_t); /* don't forget the labels */

    mnist_context->data = (float*)malloc(total_memory_requirement);
    if (mnist_context->data == NULL) {
        return 1;
    }
    mnist_context->labels = (uint8_t*)(mnist_context->data + tensor_size_from_shape(out_data_shape));


    /* Prepare the path for concatenation with the filename. Concatenation is done in open_file() */
    size_t path_size = strlen(mnist_create_info->path);
    char* file_path = (char*)malloc(path_size + 1 + strlen(dataset_info->image_filename) + 1);
    memcpy(file_path, mnist_create_info->path, path_size);
    file_path[path_size] = '/';

    uint32_t status = 0;


    /* Read the dataset */

    status = read_image_file(
        mnist_context->data,
        file_path,
        dataset_info->image_filename,
        path_size + 1,
        strlen(dataset_info->image_filename),
        dataset_info->image_file_size,
        dataset_info->num_samples,
        mnist_create_info->padding
    );

    if (status != 0) {
        goto error;
    }

    status = read_label_file(
        mnist_context->labels,
        file_path,
        dataset_info->label_filename,
        path_size + 1,
        strlen(dataset_info->label_filename),
        dataset_info->label_file_size,
        dataset_info->num_samples
    );

    if (status != 0) {
        goto error;
    }

    free(file_path);
    return 0;

error:
    free(mnist_context->data);
    free(file_path);
    return 1;
}


static uint32_t mnist_get_batch(
    dataset_context_t* context,
    const size_t* indices,
    tensor_t* out_batch,
    uint8_t* out_labels
)
{
    mnist_context_t* mnist_context = (mnist_context_t*)context;


    const size_t batch_size = tensor_shape_get_dim(tensor_get_shape(out_batch), TENSOR_BATCH_DIM);
    float* out_data = tensor_get_data(out_batch);

    for (size_t i = 0; i < batch_size; i++) {
        float* out_i = &out_data[mnist_context->sample_size * i];
        float* sample_i = &mnist_context->data[mnist_context->sample_size * indices[i]];
        memcpy(out_i, sample_i, mnist_context->sample_size * sizeof(float));
    
        out_labels[i] = mnist_context->labels[indices[i]];
    }
}


static uint32_t mnist_deinit(dataset_context_t* context)
{
    mnist_context_t* mnist_context = (mnist_context_t*)context;

    if (mnist_context->data != NULL) {
        free(mnist_context->data);
    }
}



// Struct to store a file header
typedef struct file_header {
    uint32_t magic_number;
    uint32_t number_of_items;
    uint32_t number_of_rows;
    uint32_t number_of_columns;
} file_header;


// Converts a high endian int to a low endian int
static uint32_t flip(uint32_t n)
{
    return (n >> 24 & 0xff) | (n >> 8 & 0xff00) | (n << 8 & 0xff0000) | (n << 24 & 0xff000000);
}

// Opens a file in file_path with file_name
static FILE* open_file(char* file_path, const char* file_name, size_t file_path_size, size_t file_name_size)
{
    memcpy(file_path + file_path_size, file_name, file_name_size + 1);
    return fopen(file_path, "r");
}

// Validates the file size of a file
static uint32_t validate_file_size(FILE* fp, size_t required_size)
{
    fseek(fp, 0, SEEK_END);
    int32_t file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    return file_size != required_size;
}

// Validates header data of an image file
static uint32_t validate_image_header(FILE* fp, size_t image_sample_size)
{
    file_header hdr;
    fread(&hdr, 1, 16, fp);
    hdr.magic_number = flip(hdr.magic_number);
    hdr.number_of_items = flip(hdr.number_of_items);
    hdr.number_of_rows = flip(hdr.number_of_rows);
    hdr.number_of_columns = flip(hdr.number_of_columns);
    return !(hdr.magic_number == IMAGE_MAGIC && hdr.number_of_items == image_sample_size && hdr.number_of_rows == IMAGE_HEIGHT && hdr.number_of_columns == IMAGE_WIDTH);
}

// Validates header data of a label file
static uint32_t validate_label_header(FILE* fp, size_t label_sample_size)
{
    file_header hdr;
    fread(&hdr, 1, 8, fp);
    hdr.magic_number = flip(hdr.magic_number);
    hdr.number_of_items = flip(hdr.number_of_items);
    return !(hdr.magic_number == LABEL_MAGIC && hdr.number_of_items == label_sample_size);
}

// Reads the image data out of an image file
static uint32_t read_images(FILE* fp, float* dest, size_t image_sample_size, size_t padding)
{
    const size_t image_width = IMAGE_WIDTH + 2 * padding;
    const size_t image_height = IMAGE_HEIGHT + 2 * padding;
    const size_t image_size = image_width * image_height;

    fseek(fp, 16, SEEK_SET);
    uint8_t* temp_buffer = (uint8_t*)malloc(IMAGE_SIZE * image_sample_size);
    if (!temp_buffer)
        return 1;
    fread(temp_buffer, 1, IMAGE_SIZE * image_sample_size, fp);
    memset(dest, 0, image_sample_size * image_size * sizeof(float));
    for (size_t i = 0; i < image_sample_size; i++)
        for (size_t j = 0; j < IMAGE_HEIGHT; j++)
            for (size_t k = 0; k < IMAGE_WIDTH; k++)
                dest[i * image_size + (j + padding) * image_width + k + padding] = temp_buffer[i * IMAGE_SIZE + j * IMAGE_WIDTH + k];
    
    free(temp_buffer);
    return 0;
}

// Reads an image file
static uint32_t read_image_file(float* dest, char* file_path, const char* file_name, size_t file_path_size, size_t file_name_size, size_t required_file_size, size_t sample_size, size_t padding)
{
    FILE* fp = open_file(file_path, file_name, file_path_size, file_name_size);
    if (!fp)
        return 1;
    if (validate_file_size(fp, required_file_size) || validate_image_header(fp, sample_size))
        goto error;
    read_images(fp, dest, sample_size, padding);
    fclose(fp);
    return 0;
error:
    fclose(fp);
    return 1;
}

// Reads a label file
static uint32_t read_label_file(uint8_t* dest, char* file_path, const char* file_name, size_t file_path_size, size_t file_name_size, size_t required_file_size, size_t sample_size)
{
    FILE* fp = open_file(file_path, file_name, file_path_size, file_name_size);
    if (!fp)
        return 1;
    if (validate_file_size(fp, required_file_size) || validate_label_header(fp, sample_size))
        goto error;
    fread(dest, 1, sample_size, fp);
    fclose(fp);
    return 0;
error:
    fclose(fp);
    return 1;
}
