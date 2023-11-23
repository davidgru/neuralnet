#include <malloc.h>
#include <string.h>

#include "ai_optimizer.h"


struct optimizer_s {
    optimizer_impl_t impl;
    layer_param_ref_list_t param_refs;
    void* private_data;
};


uint32_t optimizer_create(
    optimizer_t* optimizer,
    const optimizer_impl_t* impl,
    const optimizer_config_t* config
)
{
    *optimizer = (optimizer_t)malloc(sizeof(struct optimizer_s));
    if (*optimizer == NULL) {
        return 1;
    }


    (*optimizer)->impl = *impl;

    (*optimizer)->param_refs.param_refs = NULL;
    (*optimizer)->param_refs.num_params = 0;

    (*optimizer)->private_data = malloc(impl->private_data_size);
    if ((*optimizer)->private_data == NULL) {
        free(*optimizer);
        return 1;
    }
    impl->init_func((*optimizer)->private_data, config);


    return 0;
}


uint32_t optimizer_add_params(optimizer_t optimizer, layer_param_ref_list_t* refs)
{
    /* no params, so nothing to update */
    if (refs->num_params == 0) {
        return 0;
    }

    size_t new_num_params = optimizer->param_refs.num_params + refs->num_params;
    
    /* reallocate the param buffer */
    layer_param_ref_t* new_refs = (layer_param_ref_t*)calloc(new_num_params,
        sizeof(layer_param_ref_t));
    if (new_refs == NULL) {
        return 1;
    }

    if (optimizer->param_refs.param_refs != NULL) {
        /* copy old params to new buffer */
        memcpy(new_refs, optimizer->param_refs.param_refs,
            optimizer->param_refs.num_params * sizeof(layer_param_ref_t));
        /* discard old buffer*/
        free(optimizer->param_refs.param_refs);
    }

    /* copy new params to new buffer */
    memcpy(&new_refs[optimizer->param_refs.num_params], refs->param_refs,
        refs->num_params * sizeof(layer_param_ref_t));

    optimizer->param_refs.param_refs = new_refs;
    optimizer->param_refs.num_params = new_num_params;
    return 0;
}


uint32_t optimizer_step(optimizer_t optimizer)
{
    return optimizer->impl.update_func(optimizer->private_data, &optimizer->param_refs);
}


uint32_t optimizer_destroy(optimizer_t optimizer)
{
    if (optimizer->param_refs.param_refs != NULL) {
        free(optimizer->param_refs.param_refs);
    }
    free(optimizer->private_data);
    free(optimizer);
}
