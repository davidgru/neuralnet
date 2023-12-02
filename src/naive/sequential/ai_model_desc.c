
#include <stdio.h>
#include <malloc.h>
#include <memory.h>

#include "ai_model_desc.h"

#include "log.h"


/* Helper function to append any layer to the model descriptor. */
static uint32_t model_desc_add_entry(ai_model_desc_t* desc, const model_desc_entry_t* entry);



uint32_t ai_model_desc_create(ai_model_desc_t** desc)
{
    *desc = (ai_model_desc_t*)malloc(sizeof(ai_model_desc_t));
    (*desc)->num_layers = 0;
    (*desc)->entries = NULL;
    return 0;
}


uint32_t ai_model_desc_add_activation_layer(
    ai_model_desc_t* desc,
    activation_function_kind_t activation_function
)
{
    activation_layer_create_info_t* activation_create_info =
        (activation_layer_create_info_t*)malloc(sizeof(activation_layer_create_info_t));
    activation_create_info->activation_function = activation_function;
    
    model_desc_entry_t entry = {
        .layer_impl = &activation_layer_impl,
        .create_info = activation_create_info,
        .layer_kind = AI_ACTIVATION_LAYER,
    };
    return model_desc_add_entry(desc, &entry);
}


uint32_t ai_model_desc_add_convolutional_layer(
    ai_model_desc_t* desc,
    size_t output_channels,
    size_t kernel_size,
    size_t stride,
    size_t padding,
    AI_ConvLayerWeightInit weight_init,
    AI_ConvLayerBiasInit bias_init
)
{
    return ai_model_desc_add_convolutional_layer_ext(desc, output_channels,
        kernel_size, kernel_size, stride, stride, padding, padding, padding, padding, weight_init,
        bias_init);
}


uint32_t ai_model_desc_add_convolutional_layer_ext(
    ai_model_desc_t* desc,
    size_t output_channels,
    size_t kernel_height,
    size_t kernel_width,
    size_t stride_y,
    size_t stride_x,
    size_t padding_top,
    size_t padding_left,
    size_t padding_bottom,
    size_t padding_right,
    AI_ConvLayerWeightInit weight_init,
    AI_ConvLayerBiasInit bias_init
)
{
    if (padding_top != 0 || padding_left != 0 || padding_bottom != 0 || padding_right != 0) {
        LOG_ERROR("Only zero padding is supported by convolution operation.\n");
        return 1;
    }

    convolutional_layer_create_info_t* conv_create_info =
        (convolutional_layer_create_info_t*)malloc(sizeof(convolutional_layer_create_info_t));
    conv_create_info->output_channels = output_channels;
    conv_create_info->filter_height = kernel_height;
    conv_create_info->filter_width = kernel_width;
    conv_create_info->stride_y = stride_y;
    conv_create_info->stride_x = stride_x;
    conv_create_info->weight_init = weight_init;
    conv_create_info->bias_init = bias_init;

    model_desc_entry_t entry = {
        .layer_impl = &convolutional_layer_impl,
        .create_info = conv_create_info,
        .layer_kind = AI_CONVOLUTIONAL_LAYER,
    };
    return model_desc_add_entry(desc, &entry);
}


uint32_t ai_model_desc_add_dropout_layer(ai_model_desc_t* desc, float dropout_rate)
{
    dropout_layer_create_info_t* dropout_create_info =
        (dropout_layer_create_info_t*)malloc(sizeof(dropout_layer_create_info_t));
    dropout_create_info->dropout_rate = dropout_rate;

    model_desc_entry_t entry = {
        .layer_impl = &dropout_layer_impl,
        .create_info = dropout_create_info,
        .layer_kind = AI_DROPOUT_LAYER,
    };
    return model_desc_add_entry(desc, &entry);
}


uint32_t ai_model_desc_add_linear_layer(
    ai_model_desc_t* desc,
    size_t output_size,
    AI_FCLayerWeightInit weight_init,
    AI_FCLayerBiasInit bias_init
)
{
    linear_layer_create_info_t* linear_create_info =
        (linear_layer_create_info_t*)malloc(sizeof(linear_layer_create_info_t));
    linear_create_info->output_size = output_size;
    linear_create_info->weight_init = weight_init;
    linear_create_info->bias_init = bias_init;

    model_desc_entry_t entry = {
        .layer_impl = &linear_layer_impl,
        .create_info = linear_create_info,
        .layer_kind = AI_LINEAR_LAYER,
    };
    return model_desc_add_entry(desc, &entry);
}


uint32_t ai_model_desc_add_pooling_layer(
    ai_model_desc_t* desc,
    size_t kernel_size,
    size_t stride,
    size_t padding,
    pooling_kind_t pooling_kind
)
{
    return ai_model_desc_add_pooling_layer_ext(desc, kernel_size, kernel_size, stride, stride,
        padding, padding, padding, padding, pooling_kind);
}


uint32_t ai_model_desc_add_pooling_layer_ext(
    ai_model_desc_t* desc,
    size_t kernel_height,
    size_t kernel_width,
    size_t stride_y,
    size_t stride_x,
    size_t padding_top,
    size_t padding_left,
    size_t padding_bottom,
    size_t padding_right,
    pooling_kind_t pooling_kind
)
{
    if (kernel_height != kernel_width) {
        LOG_ERROR("Non-squared kernels are not supported by pooling operation.\n");
        return 1;
    }

    if (stride_x != 1 || stride_y != 1) {
        LOG_ERROR("Only stride of 1 is supported by pooling operation.\n");
        return 1;
    }

    if (padding_top != 0 || padding_left != 0 || padding_bottom != 0 || padding_right != 0) {
        LOG_ERROR("Only zero padding is supported by pooling operation.\n");
        return 1;
    }

    pooling_layer_create_info_t* pooling_create_info =
        (pooling_layer_create_info_t*)malloc(sizeof(pooling_layer_create_info_t));
    pooling_create_info->kernel_width = kernel_height;
    pooling_create_info->pooling_operation = pooling_kind;

    model_desc_entry_t entry = {
        .layer_impl = &pooling_layer_impl,
        .create_info = pooling_create_info,
        .layer_kind = AI_POOLING_LAYER,
    };
    return model_desc_add_entry(desc, &entry);
}


/**
 * @brief Print a model summary to stdout.
 * 
 * @param desc The model descriptor.
 * @return uint32_t 
 */
uint32_t ai_model_desc_dump(ai_model_desc_t* desc)
{
    printf("********************************************************************************\n");
    printf("* Printing model summary. #layers: %zu\n", desc->num_layers);
    printf("********************************************************************************\n");
    for (size_t i = 0; i < desc->num_layers; i++) {
        model_desc_entry_t* current_info = &desc->entries[i];
        switch (current_info->layer_kind) {
            case AI_ACTIVATION_LAYER:
            {
                activation_layer_create_info_t* activation_create_info =
                    (activation_layer_create_info_t*)current_info->create_info;
                printf("* activation\t(type: %d)\n", activation_create_info->activation_function);
                break;
            }
            case AI_LINEAR_LAYER:
            {
                linear_layer_create_info_t* linear_create_info = 
                    (linear_layer_create_info_t*)current_info->create_info;
                printf("* linear\t(nodes: %zu)\n", linear_create_info->output_size);
                break;
            }
            case AI_CONVOLUTIONAL_LAYER:
            {
                convolutional_layer_create_info_t* conv_create_info = 
                    (convolutional_layer_create_info_t*)current_info->create_info;
                printf("* conv\t\t(filters: %d, kernel: (%zu,%zu), stride: (%zu,%zu)" 
                    ", padding: (%zu,%zu,%zu,%zu))\n", conv_create_info->output_channels,
                    conv_create_info->filter_height, conv_create_info->filter_width,
                    conv_create_info->stride_y, conv_create_info->stride_x, 0, 0, 0,
                    0);
                break;
            }
            case AI_POOLING_LAYER:
            {
                pooling_layer_create_info_t* pooling_create_info = 
                    (pooling_layer_create_info_t*)current_info->create_info;
                printf("* pooling\t(kernel: (%zu,%zu), algorithm: %d)\n",
                    pooling_create_info->kernel_width, pooling_create_info->kernel_width,
                    pooling_create_info->pooling_operation);
                break;
            }
            case AI_DROPOUT_LAYER:
            {
                dropout_layer_create_info_t* dropout_create_info =
                    (dropout_layer_create_info_t*)current_info->create_info;
                printf("* dropout\t(rate: %f)\n", dropout_create_info->dropout_rate);
                break;
            }
            default:
            {
                printf("* unknown\n");
                break;
            }
        }
    }
    printf("********************************************************************************\n");
}


uint32_t ai_model_desc_destroy(ai_model_desc_t* desc)
{
    for (size_t i = 0; i < desc->num_layers; i++) {
        free(desc->entries[i].create_info);
    }
    free(desc->entries);
    free(desc);
    return 0;
}


static uint32_t model_desc_add_entry(ai_model_desc_t* desc, const model_desc_entry_t* entry)
{
    /* reallocate and copy to grow dynamically */
    model_desc_entry_t* new_entries = (model_desc_entry_t*)calloc(desc->num_layers + 1, sizeof(model_desc_entry_t));
    if (desc->entries != NULL) {
        memcpy(new_entries, desc->entries, desc->num_layers * sizeof(model_desc_entry_t));
        free(desc->entries);
    }
    desc->entries = new_entries;

    /* add new create info at the end */
    memcpy(&desc->entries[desc->num_layers], entry, sizeof(model_desc_entry_t));
    desc->num_layers++;

    return 0;
}
