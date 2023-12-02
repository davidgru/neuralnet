
#include "model_desc.h"

#include <stdio.h>
#include <malloc.h>
#include <memory.h>

static uint32_t model_desc_add_create_info(model_desc_t* desc, dnnl_layer_create_info_t* create_info)
{

    // Allocate bigger memory and copy all elements to the new memory
    dnnl_layer_create_info_t* new = (dnnl_layer_create_info_t*)malloc(sizeof(dnnl_layer_create_info_t) * (desc->num_layers + 1));
    if (desc->create_infos) {
        memcpy(new, desc->create_infos, sizeof(dnnl_layer_create_info_t) * desc->num_layers);
        free(desc->create_infos);
    }
    desc->create_infos = new;

    // Copy the new create info to the last index of the list
    memcpy(&desc->create_infos[desc->num_layers], create_info, sizeof(dnnl_layer_create_info_t));

    desc->num_layers++;

    return 0;
}


uint32_t model_desc_create(model_desc_t** desc, input_dims_t* input_shape, dnnl_loss_kind_t loss)
{
    *desc = (model_desc_t*)malloc(sizeof(model_desc_t));

    (*desc)->num_layers = 0;
    (*desc)->create_infos = 0;
    (*desc)->loss = loss;
    memcpy(&(*desc)->input_shape, input_shape, sizeof(input_dims_t));

    return 0;
}

uint32_t model_desc_dump(model_desc_t* desc)
{
    printf("model size: %d layers\n", desc->num_layers);
    printf("input shape: %d, %d, %d, %d\n", desc->input_shape.N, desc->input_shape.C, desc->input_shape.H, desc->input_shape.W);
    for (size_t i = 0; i < desc->num_layers; i++) {
        switch (desc->create_infos[i].layer_kind) {
            case dnnl_layer_kind_activation:
                printf("activation layer\n");
                break;
            case dnnl_layer_kind_linear:
                printf("linear layer\n");
                break;
            case dnnl_layer_kind_convolutional:
                printf("convolutional_layer\n");
                break;
            case dnnl_layer_kind_pooling:
                printf("pooling layer\n");
                break;
        }
    }

    switch(desc->loss) {
        case dnnl_loss_mse:
            printf("mse loss\n");
            break;
    }

    return 0;
}

uint32_t model_desc_add_activation_layer(model_desc_t* desc, dnnl_activation_kind_t kind)
{
    dnnl_activation_layer_create_info_t* ai = (dnnl_activation_layer_create_info_t*)malloc(sizeof(dnnl_activation_layer_create_info_t));
    ai->activation = kind;
    ai->allow_reorder = 1;
    dnnl_layer_create_info_t i;
    i.layer_kind = dnnl_layer_kind_activation;
    i.layer_create_info = ai;
    model_desc_add_create_info(desc, &i);
    return 0;
}

uint32_t model_desc_add_linear_layer(model_desc_t* desc, size_t OC, dnnl_linear_layer_weight_init_kind_t weight_init, dnnl_linear_layer_bias_init_kind_t bias_init)
{
    dnnl_linear_layer_create_info_t* li = (dnnl_linear_layer_create_info_t*)malloc(sizeof(dnnl_linear_layer_create_info_t));
    li->OC = OC;
    li->weight_init = weight_init;
    li->bias_init = bias_init;
    dnnl_layer_create_info_t i;
    i.layer_kind = dnnl_layer_kind_linear;
    i.layer_create_info = li;
    model_desc_add_create_info(desc, &i);
    return 0;
}

uint32_t model_desc_add_convolutional_layer(model_desc_t* desc, float learning_rate, size_t OC, size_t KH, size_t KW, size_t SH, size_t SW, size_t PT, size_t PL, size_t PB, size_t PR, dnnl_convolutional_layer_weight_init_kind_t weight_init, dnnl_convolutional_layer_bias_init_kind_t bias_init)
{
    dnnl_convolutional_layer_create_info_t* ci = (dnnl_convolutional_layer_create_info_t*)malloc(sizeof(dnnl_convolutional_layer_create_info_t));
    ci->OC = OC;
    ci->KH = KH;
    ci->KW = KW;
    ci->SH = SH;
    ci->SW = SW;
    ci->PT = PT;
    ci->PL = PL;
    ci->PB = PB;
    ci->PR = PR;
    ci->weight_init = weight_init;
    ci->bias_init = bias_init;
    ci->learning_rate = learning_rate;
    dnnl_layer_create_info_t i;
    i.layer_kind = dnnl_layer_kind_convolutional;
    i.layer_create_info = ci;
    model_desc_add_create_info(desc, &i);
    return 0;
}

uint32_t model_desc_add_pooling_layer(model_desc_t* desc, size_t KH, size_t KW, size_t SH, size_t SW, size_t PT, size_t PL, size_t PB, size_t PR, dnnl_pooling_kind_t kind)
{
    dnnl_pooling_layer_create_info_t* pi = (dnnl_pooling_layer_create_info_t*)malloc(sizeof(dnnl_pooling_layer_create_info_t));
    pi->KH = KH;
    pi->KW = KW;
    pi->SH = SH;
    pi->SW = SW;
    pi->PT = PT;
    pi->PL = PL;
    pi->PB = PB;
    pi->PR = PR;
    pi->pooling_kind = kind;
    dnnl_layer_create_info_t i;
    i.layer_kind = dnnl_layer_kind_pooling;
    i.layer_create_info = pi;
    model_desc_add_create_info(desc, &i);
    return 0;
}

uint32_t model_desc_destroy(model_desc_t* desc)
{
    for (size_t i = 0; i < desc->num_layers; i++)
        free(desc->create_infos[i].layer_create_info);
    free(desc->create_infos);
    free(desc);

    return 0;
}
