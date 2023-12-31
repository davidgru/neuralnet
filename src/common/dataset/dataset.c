#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "dataset.h"
#include "tensor_impl.h"


#include "util/ai_math.h"


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

    bool normalize;
    dataset_statistics_t statistics;
};


static void shuffle_array(size_t* array, size_t size);
static void calc_dataset_statistics(dataset_t dataset, float* out_mean, float* out_stddev);


uint32_t dataset_create(
    dataset_t* dataset,
    const dataset_impl_t* impl,
    const dataset_create_info_t* create_info,
    bool normalize,
    /* can be null. in this case calculate statistics */
    const dataset_statistics_t* statistics
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

    (*dataset)->normalize = normalize;
    if (normalize) {
        if (statistics == NULL) {
            calc_dataset_statistics(*dataset, &(*dataset)->statistics.mean,
                &(*dataset)->statistics.stddev);
        } else {
            (*dataset)->statistics = *statistics;
        }
    }

    return 0;
}


const tensor_shape_t* dataset_get_shape(dataset_t dataset)
{
    return &dataset->data_shape;
}


const dataset_statistics_t* dataset_get_statistics(dataset_t dataset)
{
    if (!dataset->normalize) {
        calc_dataset_statistics(dataset, &dataset->statistics.mean,
                &dataset->statistics.stddev);
    }
    return &dataset->statistics;
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

        tensor_shape_t scratch_shape = make_tensor_shape(
            TENSOR_MAX_DIMS,
            batch_size,
            tensor_shape_get_dim(&dataset->data_shape, TENSOR_CHANNEL_DIM),
            tensor_shape_get_dim(&dataset->data_shape, TENSOR_HEIGHT_DIM),
            tensor_shape_get_dim(&dataset->data_shape, TENSOR_WIDTH_DIM)
        );
        tensor_allocate(&dataset->scratch, &scratch_shape);
    
        dataset->current_out_labels = malloc(batch_size);
    }

    const size_t num_samples = tensor_shape_get_dim(&dataset->data_shape, TENSOR_BATCH_DIM);
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
    const size_t num_samples = tensor_shape_get_dim(&dataset->data_shape, TENSOR_BATCH_DIM);

    /* The iteration is complete. */
    if (dataset->current == num_samples) {
        if (out_batch != NULL) {
            *out_batch = NULL;
        }
        if (out_labels != NULL) {
            *out_labels = NULL;
        }
        return 0;
    }

    /* Can potentially have partial batches when the batch_size is not a multiple of the dataset
        size. */
    size_t this_batch_size = dataset->batch_size;
    if (num_samples - dataset->current < dataset->batch_size) {
        this_batch_size = num_samples - dataset->current;
    }

    tensor_t* o_batch;
    uint8_t* o_labels;
    if (this_batch_size == dataset->batch_size) {
        dataset->impl->get_batch_func(dataset->impl_context, &dataset->ordering[dataset->current],
            &dataset->scratch, dataset->current_out_labels);
        
        o_batch = &dataset->scratch;
        o_labels = dataset->current_out_labels;
    } else {
        /* Reflect specific batch size in the output tensor and use scratch mem as buffer. */
        /* Will cause memory allocation when using onednn :( */
        tensor_shape_t current_out_shape = make_tensor_shape(
            TENSOR_MAX_DIMS,
            this_batch_size,
            tensor_shape_get_dim(&dataset->data_shape, TENSOR_CHANNEL_DIM),
            tensor_shape_get_dim(&dataset->data_shape, TENSOR_HEIGHT_DIM),
            tensor_shape_get_dim(&dataset->data_shape, TENSOR_WIDTH_DIM)
        );

        tensor_from_memory(&dataset->current_out_batch, &current_out_shape,
            tensor_get_data(&dataset->scratch));
        
        dataset->impl->get_batch_func(dataset->impl_context, &dataset->ordering[dataset->current],
            &dataset->current_out_batch, dataset->current_out_labels);
        
        o_batch = &dataset->current_out_batch;
        o_labels = dataset->current_out_labels;
    }

    if (dataset->normalize) {
        float* data = tensor_get_data(o_batch);
        const size_t numelem = tensor_size_from_shape(tensor_get_shape(o_batch));
        VectorAddScalar(data, -dataset->statistics.mean, numelem);
        VectorScale(data, 1.0f / dataset->statistics.stddev, numelem);
    }

    if (out_batch != NULL) {
        *out_batch = o_batch;
    }
    if (out_labels != NULL) {
        *out_labels = o_labels;
    }

    dataset->current += this_batch_size;

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


static void calc_dataset_statistics(dataset_t dataset, float* out_mean, float* out_stddev)
{
    const size_t dataset_size = tensor_shape_get_dim(&dataset->data_shape, TENSOR_BATCH_DIM);
    tensor_t* out_batch;
    float mean;
    float stddev;

    /* need to set dataset mean and stddev because in the following iterations the
        normalization is applied already. */
    dataset->statistics.mean = 0.0f;
    dataset->statistics.stddev = 1.0f;

    /* first iteration for mean calculation */
    out_batch = NULL;
    mean = 0.0f;
    dataset_iteration_begin(dataset, 1, false, &out_batch, NULL);
    while (out_batch != NULL) {
        float* data = tensor_get_data(out_batch);
        mean += Mean(data, tensor_size_from_shape(tensor_get_shape(out_batch)));
        dataset_iteration_next(dataset, &out_batch, NULL);
    }
    mean /= dataset_size;

    /* second iteration for variance calculation */
    out_batch = NULL;
    dataset_iteration_begin(dataset, 1, false, &out_batch, NULL);
    float var = 0.0f;
    while (out_batch != NULL) {
        float batch_var = 0.0f;
        const float* data = tensor_get_data_const(out_batch);
        const size_t numelem = tensor_size_from_shape(tensor_get_shape(out_batch));
        for (size_t i = 0; i < numelem; i++) {
            batch_var += (data[i] - mean) * (data[i] - mean);
        }
        batch_var /= numelem;
        var += batch_var;
        dataset_iteration_next(dataset, &out_batch, NULL);
    }
    var /= dataset_size;
    stddev = sqrtf(var);

    if (out_mean != NULL) {
        *out_mean = mean;
    }

    if (out_stddev != NULL) {
        *out_stddev = stddev;
    }
}
