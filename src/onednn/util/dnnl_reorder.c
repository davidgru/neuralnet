#include "context_impl.h"
#include "log.h"

#include "dnnl_reorder.h"


dnnl_status_t dnnl_reorder_create(
    dnnl_reorder_t* reorder,
    const_dnnl_memory_desc_t src_md,
    const_dnnl_memory_desc_t dst_md
)
{
    dnnl_status_t status = dnnl_success;
    dnnl_engine_t engine = get_dnnl_engine();

    if (dnnl_memory_desc_equal(src_md, dst_md)) {
        /* No need for reorder. */
        reorder->primitive = NULL;
    } else {
        /* Need to set up reorder primitive and output memory */

        dnnl_primitive_desc_t pd;
        status = dnnl_reorder_primitive_desc_create(&pd, src_md, engine, dst_md, engine, NULL);
        if (status != dnnl_success) {
            LOG_ERROR("Creating reorder pd failed with code %d\n", status);
            return status;
        }

        status = dnnl_primitive_create(&reorder->primitive, pd);
        if (status != dnnl_success) {
            LOG_ERROR("Creating reorder primitive failed with code %d\n", status);
            dnnl_primitive_desc_destroy(pd);
            return status;
        }
        dnnl_primitive_desc_destroy(pd);

        if (tensor_from_desc(&reorder->output, dst_md, DNNL_MEMORY_ALLOCATE) != 0){
            LOG_ERROR("Allocating reorder output memory failed\n");
            return dnnl_out_of_memory;
        }
    }

    return status;
}


dnnl_status_t dnnl_reorder_execute(
    dnnl_reorder_t* reorder,
    const tensor_t* input,
    const tensor_t** output
)
{
    dnnl_status_t status = dnnl_success;

    if (reorder->primitive == NULL) {
        /* No need for a reorder, simply forward input */
        *output = input;
    } else {
        dnnl_stream_t stream = get_dnnl_stream();
        dnnl_exec_arg_t args[] = {
            {DNNL_ARG_FROM, input->mem},
            {DNNL_ARG_TO, reorder->output.mem},
        };

        dnnl_status_t status = dnnl_primitive_execute(reorder->primitive, stream,
            sizeof(args) / sizeof(*args), args);
        if (status != dnnl_success) {
            LOG_ERROR("Reorder execution failed with code %d\n", status);
        }
        *output = &reorder->output;
    }

    return status;
}


dnnl_status_t dnnl_reorder_destroy(dnnl_reorder_t* reorder)
{
    if (reorder->primitive != NULL) {
        dnnl_primitive_destroy(reorder->primitive);
        tensor_destory(&reorder->output);
    }
    return dnnl_success;
}
