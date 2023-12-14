#include <stdlib.h>
#include <string.h>

#include "dataset.h"
#include "tensor_impl.h"


struct dataset_s {
    const dataset_impl_t* impl;
    dataset_context_t* impl_context;

    tensor_shape_t data_shape;
    /* used to store the current tensor during iteration */
    tensor_t scratch;
    tensor_t current_out_batch;
    uint8_t* current_out_labels;
    /* specifies the iteration order if shuffle = true */
    size_t* ordering;
    /* current iteration index */
    size_t current;
    size_t batch_size;
    bool shuffle;
};


static void shuffle_array(size_t* array, size_t size);


uint32_t dataset_create(
    dataset_t* dataset,
    const dataset_impl_t* impl,
    const dataset_create_info_t* create_info
)
{
    *dataset = (dataset_t)malloc(sizeof(struct dataset_s));
    if (dataset == NULL) {
        *dataset = NULL;
        return 1;
    }

    (*dataset)->impl = impl;

    (*dataset)->impl_context = (dataset_context_t*)malloc(impl->conext_size);
    if ((*dataset)->impl_context == NULL) {
        free(*dataset);
        *dataset = NULL;
        return 1;
    }
    if (impl->init_func((*dataset)->impl_context, create_info, &(*dataset)->data_shape) != 0) {
        free((*dataset)->impl_context);
        free(*dataset);
        *dataset = NULL;
        return 1;
    }

    (*dataset)->batch_size = 0;
    (*dataset)->current = 0;
    (*dataset)->shuffle = false;
    (*dataset)->current_out_labels = NULL;
    (*dataset)->ordering = NULL;

    return 0;
}


const tensor_shape_t* dataset_get_shape(dataset_t dataset)
{
    return &dataset->data_shape;
}


uint32_t dataset_iteration_begin(
    dataset_t dataset,
    size_t batch_size,
    bool shuffle,
    tensor_t** out_batch,
    uint8_t** out_labels
)
{
    if (dataset->batch_size != batch_size) {
        /* (re-)allocate the scratch buffer to match the batch_size */
        if (dataset->current_out_labels != NULL) {
            tensor_destory(&dataset->scratch);
            free(dataset->current_out_labels);
        }

        tensor_shape_t scratch_shape = dataset->data_shape;
        scratch_shape.dims[TENSOR_BATCH_DIM] = batch_size;
        tensor_allocate(&dataset->scratch, &scratch_shape);
    
        dataset->current_out_labels = (uint8_t*)malloc(batch_size);
    }

    const size_t num_samples = dataset->data_shape.dims[TENSOR_BATCH_DIM];
    if (dataset->ordering == NULL) {
        dataset->ordering = (size_t*)calloc(num_samples, sizeof(size_t));
        for (size_t i = 0; i < num_samples; i++) {
            dataset->ordering[i] = i;
        }
    }

    if (shuffle) {
        shuffle_array(dataset->ordering, num_samples);
    }


    dataset->current = 0;
    dataset->batch_size = batch_size;
    dataset->shuffle = shuffle;

    return dataset_iteration_next(dataset, out_batch, out_labels);
}


uint32_t dataset_iteration_next(dataset_t dataset, tensor_t** out_batch, uint8_t** out_labels)
{
    const size_t num_samples = dataset->data_shape.dims[TENSOR_BATCH_DIM];

    /* The iteration is complete. */
    if (dataset->current == num_samples) {
        *out_batch = NULL;
        *out_labels = NULL;
        return 0;
    }

    /* Can potentially have partial batches when the batch_size is not a multiple of the dataset
        size. */
    size_t this_batch_size = dataset->batch_size;
    if (num_samples - dataset->current < dataset->batch_size) {
        this_batch_size = num_samples - dataset->current;
    }

    /* Reflect specific batch size in the output tensor and use previously scratch mem as buffer. */
    tensor_shape_t current_out_shape = *tensor_get_shape(&dataset->scratch);
    current_out_shape.dims[TENSOR_BATCH_DIM] = this_batch_size;
    tensor_from_memory(&dataset->current_out_batch, &current_out_shape,
        tensor_get_data(&dataset->scratch));

    dataset->impl->get_batch_func(dataset->impl_context, &dataset->ordering[dataset->current],
        &dataset->current_out_batch, dataset->current_out_labels);
    
    dataset->current += this_batch_size;

    *out_batch = &dataset->current_out_batch;
    *out_labels = dataset->current_out_labels;

    return num_samples - dataset->current;
}


uint32_t dataset_destroy(dataset_t dataset)
{
    if (dataset != NULL) {
        if (dataset->impl->deinit_func != NULL) {
            dataset->impl->deinit_func(dataset->impl_context);
        }
        tensor_destory(&dataset->scratch);
        if (dataset->current_out_labels != NULL) {
            free(dataset->current_out_labels);
        }
        free(dataset->impl_context);
        if (dataset->ordering != NULL) {
            free(dataset->ordering);
        }
        free(dataset);
    }
}


static void shuffle_array(size_t* array, size_t size)
{
    if (size <= 1) {
        return;
    }
    for (size_t i = size - 1; i > 0; i--) {
        size_t j = (size_t)((double)(rand() / RAND_MAX) * (i + 1));
        /* Not sure if it can happen, but be safe. */
        if (j == i + 1) {
            j = i;
        }
        size_t tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
}
