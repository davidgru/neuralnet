
#include "ai_mnist.h"


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <windows.h>

#define TRAIN_IMAGE_SAMPLE_SIZE 60000
#define TEST_IMAGE_SAMPLE_SIZE 10000
#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT)

#define IMAGE_MAGIC 2051
#define LABEL_MAGIC 2049

#define TRAIN_IMAGE_FILE_NAME "train-images-idx3-ubyte"
#define TEST_IMAGE_FILE_NAME "t10k-images-idx3-ubyte"
#define TRAIN_LABEL_FILE_NAME "train-labels-idx1-ubyte"
#define TEST_LABEL_FILE_NAME "t10k-labels-idx1-ubyte"

#define TRAIN_IMAGE_FILE_NAME_LENGTH    (sizeof(TRAIN_IMAGE_FILE_NAME) - 1)
#define TEST_IMAGE_FILE_NAME_LENGTH     (sizeof(TEST_IMAGE_FILE_NAME) - 1)
#define TRAIN_LABEL_FILE_NAME_LENGTH    (sizeof(TRAIN_LABEL_FILE_NAME) - 1)
#define TEST_LABEL_FILE_NAME_LENGTH     (sizeof(TEST_LABEL_FILE_NAME) - 1)

#define TRAIN_IMAGE_FILE_SIZE   47040016
#define TEST_IMAGE_FILE_SIZE     7840016
#define TRAIN_LABEL_FILE_SIZE      60008
#define TEST_LABEL_FILE_SIZE       10008

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
                dest[i * image_size + (j + padding) * image_width + k + padding] = temp_buffer[i * IMAGE_SIZE + j * IMAGE_WIDTH + k] / 127.5f - 1.0f;
    
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

// Loads the mnist dataset from the specified path on disk
uint32_t AI_MnistDatasetLoad(AI_MnistDataset* dataset, const char* path, size_t padding)
{
    dataset->num_train_images = TRAIN_IMAGE_SAMPLE_SIZE;
    dataset->num_test_images = TEST_IMAGE_SAMPLE_SIZE;
    dataset->image_width = IMAGE_WIDTH + 2 * padding;
    dataset->image_height = IMAGE_HEIGHT + 2 * padding;

    const size_t image_size = dataset->image_width * dataset->image_height;

    // Allocate memory
    size_t total_dataset_size = (TRAIN_IMAGE_SAMPLE_SIZE + TEST_IMAGE_SAMPLE_SIZE) * image_size * sizeof(float)
        + (TRAIN_IMAGE_SAMPLE_SIZE + TEST_IMAGE_SAMPLE_SIZE) * sizeof(uint8_t);
    
    dataset->train_images = (float*)malloc(total_dataset_size);
    if (!dataset->train_images) return 1;
    dataset->test_images = dataset->train_images + TRAIN_IMAGE_SAMPLE_SIZE * image_size;
    dataset->train_labels = (uint8_t*)(dataset->test_images + TEST_IMAGE_SAMPLE_SIZE * image_size);
    dataset->test_labels = dataset->train_labels + TRAIN_IMAGE_SAMPLE_SIZE;

    /* Read the dataset */

    size_t path_size = strlen(path);
    char* file_path = (char*)malloc(path_size + 1 + TRAIN_IMAGE_FILE_NAME_LENGTH + 1);
    memcpy(file_path, path, path_size);
    file_path[path_size] = '/';

    uint32_t status = 0;


    /* Read train images */

    status = read_image_file(
        dataset->train_images,
        file_path,
        TRAIN_IMAGE_FILE_NAME,
        path_size + 1,
        TRAIN_IMAGE_FILE_NAME_LENGTH,
        TRAIN_IMAGE_FILE_SIZE,
        TRAIN_IMAGE_SAMPLE_SIZE,
        padding
    );
    if (status)
        goto error;

    /* Read test images */

    status = read_image_file(
        dataset->test_images,
        file_path,
        TEST_IMAGE_FILE_NAME,
        path_size + 1,
        TEST_IMAGE_FILE_NAME_LENGTH,
        TEST_IMAGE_FILE_SIZE,
        TEST_IMAGE_SAMPLE_SIZE,
        padding
    );
    if (status)
        goto error;
    
    /* Read train labels */

    status = read_label_file(
        dataset->train_labels,
        file_path,
        TRAIN_LABEL_FILE_NAME,
        path_size + 1,
        TRAIN_LABEL_FILE_NAME_LENGTH,
        TRAIN_LABEL_FILE_SIZE,
        TRAIN_IMAGE_SAMPLE_SIZE
    );
    if (status)
        goto error;

    /* Read test labels */

    status = read_label_file(
        dataset->test_labels,
        file_path,
        TEST_LABEL_FILE_NAME,
        path_size + 1,
        TEST_LABEL_FILE_NAME_LENGTH,
        TEST_LABEL_FILE_SIZE,
        TEST_IMAGE_SAMPLE_SIZE
    );
    if (status)
        goto error;


    free(file_path);
    return 0;

error:
    free(dataset->train_images);
    free(file_path);
    return 1;
}

// Destroys a mnist dataset
void AI_MnistDatasetFree(AI_MnistDataset* dataset)
{
    if (dataset->train_images)
        free(dataset->train_images);
}
