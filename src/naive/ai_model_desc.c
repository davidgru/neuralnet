
#include <stdio.h>
#include <malloc.h>
#include <memory.h>

#include "ai_model_desc.h"

#include "log.h"

#include "layer/ai_convolutional_layer.h"


/* Helper function to append any layer to the model descriptor. */
static uint32_t model_desc_add_create_info(ai_model_desc_t* desc, AI_LayerCreateInfo* create_info);



uint32_t ai_model_desc_create(ai_model_desc_t** desc)
{
    *desc = (ai_model_desc_t*)malloc(sizeof(ai_model_desc_t));
    (*desc)->num_layers = 0;
    (*desc)->create_infos = 0;
    return 0;
}


uint32_t ai_model_desc_add_activation_layer(
    ai_model_desc_t* desc,
    AI_ActivationFunctionEnum activation_function
)
{
    AI_ActivationLayerCreateInfo* activation_create_info =
        (AI_ActivationLayerCreateInfo*)malloc(sizeof(AI_ActivationLayerCreateInfo));
    activation_create_info->activation_function = activation_function;
    
    AI_LayerCreateInfo create_info = {
        .type = AI_ACTIVATION_LAYER,
        .create_info = (void*)activation_create_info,
    };
    return model_desc_add_create_info(desc, &create_info);
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

    AI_LayerCreateInfo create_info = {
        .type = AI_CONVOLUTIONAL_LAYER,
        .create_info = (void*)conv_create_info
    };
    return model_desc_add_create_info(desc, &create_info);
}


uint32_t ai_model_desc_add_dropout_layer(ai_model_desc_t* desc, float dropout_rate)
{
    AI_DropoutLayerCreateInfo* dropout_create_info =
        (AI_DropoutLayerCreateInfo*)malloc(sizeof(AI_DropoutLayerCreateInfo));
    dropout_create_info->dropout_rate = dropout_rate;

    AI_LayerCreateInfo create_info = {
        .type = AI_DROPOUT_LAYER,
        .create_info = (void*)dropout_create_info
    };
    return model_desc_add_create_info(desc, &create_info);
}


uint32_t ai_model_desc_add_linear_layer(
    ai_model_desc_t* desc,
    size_t output_size,
    AI_FCLayerWeightInit weight_init,
    AI_FCLayerBiasInit bias_init
)
{
    AI_LinearLayerCreateInfo* linear_create_info =
        (AI_LinearLayerCreateInfo*)malloc(sizeof(AI_LinearLayerCreateInfo));
    linear_create_info->output_size = output_size;
    linear_create_info->weight_init = weight_init;
    linear_create_info->bias_init = bias_init;

    AI_LayerCreateInfo create_info = {
        .type = AI_LINEAR_LAYER,
        .create_info = (void*)linear_create_info
    };
    return model_desc_add_create_info(desc, &create_info);
}


uint32_t ai_model_desc_add_pooling_layer(
    ai_model_desc_t* desc,
    size_t kernel_size,
    size_t stride,
    size_t padding,
    AI_PoolingOperationEnum pooling_kind
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
    AI_PoolingOperationEnum pooling_kind
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

    AI_PoolingLayerCreateInfo* pooling_create_info =
        (AI_PoolingLayerCreateInfo*)malloc(sizeof(AI_PoolingLayerCreateInfo));
    pooling_create_info->kernel_width = kernel_height;
    pooling_create_info->pooling_operation = pooling_kind;

    AI_LayerCreateInfo create_info = {
        .type = AI_POOLING_LAYER,
        .create_info = (void*)pooling_create_info
    };
    return model_desc_add_create_info(desc, &create_info);
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
        AI_LayerCreateInfo* current_info = &desc->create_infos[i];
        switch (current_info->type) {
            case AI_ACTIVATION_LAYER:
            {
                AI_ActivationLayerCreateInfo* activation_create_info =
                    (AI_ActivationLayerCreateInfo*)current_info->create_info;
                printf("* activation\t(type: %d)\n", activation_create_info->activation_function);
                break;
            }
            case AI_LINEAR_LAYER:
            {
                AI_LinearLayerCreateInfo* linear_create_info = 
                    (AI_LinearLayerCreateInfo*)current_info->create_info;
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
                AI_PoolingLayerCreateInfo* pooling_create_info = 
                    (AI_PoolingLayerCreateInfo*)current_info->create_info;
                printf("* pooling\t(kernel: (%zu,%zu), algorithm: %d)\n",
                    pooling_create_info->kernel_width, pooling_create_info->kernel_width,
                    pooling_create_info->pooling_operation);
                break;
            }
            case AI_DROPOUT_LAYER:
            {
                AI_DropoutLayerCreateInfo* dropout_create_info =
                    (AI_DropoutLayerCreateInfo*)current_info->create_info;
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
    for (size_t i = 0; i < desc->num_layers; i++)
        free(desc->create_infos[i].create_info);
    free(desc->create_infos);
    free(desc);
    return 0;
}


static uint32_t model_desc_add_create_info(ai_model_desc_t* desc, AI_LayerCreateInfo* create_info)
{
    /* reallocate and copy to grow dynamically */
    AI_LayerCreateInfo* new_create_infos = (AI_LayerCreateInfo*)malloc(sizeof(AI_LayerCreateInfo) * (desc->num_layers + 1));
    if (desc->create_infos) {
        memcpy(new_create_infos, desc->create_infos, sizeof(AI_LayerCreateInfo) * desc->num_layers);
        free(desc->create_infos);
    }
    desc->create_infos = new_create_infos;

    /* add new create info at the end */
    memcpy(&desc->create_infos[desc->num_layers], create_info, sizeof(AI_LayerCreateInfo));
    desc->num_layers++;

    return 0;
}
