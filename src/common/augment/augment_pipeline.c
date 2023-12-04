#include <stdbool.h>
#include <stdlib.h>

#include "augment_pipeline.h"
#include "tensor_impl.h"
#include "random.h"


struct augment_pipeline_s {
    size_t num_impls;
    tensor_t scratch_mem;
    tensor_t output;
    const augment_impl_t** impls;
    augment_context_t** contexts;
    
    bool tensor_allocated;
};


uint32_t augment_pipeline_create(
    augment_pipeline_t* pipeline,
    const augment_pipeline_config_t* config
)
{
    *pipeline = malloc(sizeof(**pipeline));
    if (*pipeline == NULL) {
        return 1;
    }


    (*pipeline)->impls = NULL;
    (*pipeline)->contexts = NULL;
    (*pipeline)->tensor_allocated = false;


    (*pipeline)->num_impls = config->num_entries;
    (*pipeline)->impls = calloc(config->num_entries, sizeof(*(*pipeline)->impls));
    if ((*pipeline)->impls == NULL) {
        augment_pipeline_destroy(*pipeline);
        return 1;
    }


    (*pipeline)->contexts = calloc(config->num_entries, sizeof(*(*pipeline)->contexts));
    if ((*pipeline)->contexts == NULL) {
        augment_pipeline_destroy(*pipeline);
        return 1;
    }


    for (size_t i = 0; i < config->num_entries; i++) {
        (*pipeline)->impls[i] = config->entries[i].impl;
        
        /* alloc a context if needed */
        if ((*pipeline)->impls[i]->context_size != 0) {
            (*pipeline)->contexts[i] = malloc((*pipeline)->impls[i]->context_size);
            if ((*pipeline)->contexts[i] == NULL) {
                augment_pipeline_destroy(*pipeline);
                return 1;
            }
        } else {
            (*pipeline)->contexts[i] = NULL;
        }

        /* initialize the augmentation step */
        (*pipeline)->impls[i]->init_func((*pipeline)->contexts[i], config->entries[i].config);
    }

    return 0;
}


uint32_t augment_pipeline_forward(
    augment_pipeline_t pipeline,
    const tensor_t* input,
    tensor_t** out_output
)
{
    const augment_impl_t* impl = pipeline->impls[0];
    augment_inplace_func_t inplace_func = impl->augment_inplace_func;


    const tensor_shape_t* input_shape = tensor_get_shape(input);

    if (!pipeline->tensor_allocated) {
        tensor_allocate(&pipeline->scratch_mem, input_shape);
        pipeline->tensor_allocated = true;
    }

    /* Construct output tensor with input batch size and embed into scratch_mem */
    tensor_shape_t output_shape = *tensor_get_shape(&pipeline->scratch_mem);
    output_shape.dims[TENSOR_BATCH_DIM] = input_shape->dims[TENSOR_BATCH_DIM];
    tensor_from_memory(&pipeline->output, &output_shape, tensor_get_data(&pipeline->scratch_mem));


    tensor_copy(&pipeline->output, input);
    for (size_t i = 0; i < pipeline->num_impls; i++) {
        pipeline->impls[i]->augment_inplace_func(pipeline->contexts[i], &pipeline->output);
    }

    if (out_output != NULL) {
        *out_output = &pipeline->output;
    }

}


uint32_t augment_pipeline_destroy(augment_pipeline_t pipeline)
{
    if (pipeline->tensor_allocated) {
        tensor_destory(&pipeline->scratch_mem);
    }

    if (pipeline->impls != NULL) {
        for (size_t i = 0; i < pipeline->num_impls; i++) {
            pipeline->impls[i]->deinit_func(pipeline->contexts[i]);
        }
        free(pipeline->impls);
    }

    if (pipeline->contexts != NULL) {
        for (size_t i = 0; i < pipeline->num_impls; i++) {
            if (pipeline->contexts[i] != NULL) {
                free(pipeline->contexts[i]);
            }
        }
        free(pipeline->contexts);
    }

    free(pipeline);
}
