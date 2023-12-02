
#include "dnnl_reorder.h"

#define CHECK_DNNL(call) if (call) goto dnnl_error

dnnl_status_t dnnl_reorder_create(dnnl_reorder_t* reorder, dnnl_memory_t src_mem, const dnnl_memory_desc_t* dst_md)
{
    reorder->src_mem = src_mem;

    // Get src memory desc
    const dnnl_memory_desc_t* src_md;
    CHECK_DNNL(dnnl_memory_get_memory_desc(src_mem, &src_md));

    // Get the src memory engine
    dnnl_engine_t engine;
    CHECK_DNNL(dnnl_memory_get_engine(src_mem, &engine));

    // Compare src and dst memory descs
    reorder->need_reorder = !dnnl_memory_desc_equal(src_md, dst_md);

    if (reorder->need_reorder) {
        // Create reorder primitive if necessary
        dnnl_primitive_desc_t pd;
        CHECK_DNNL(dnnl_reorder_primitive_desc_create(&pd, src_md, engine, dst_md, engine, 0));
        CHECK_DNNL(dnnl_primitive_create(&reorder->primitive, pd));
        // Create dst memory if necessary
        CHECK_DNNL(dnnl_memory_create(&reorder->dst_mem, dst_md, engine, DNNL_MEMORY_ALLOCATE));
    }
    else
        reorder->dst_mem = reorder->src_mem;

    return 0;
dnnl_error:
    return 1;
}

dnnl_status_t dnnl_reorder_execute(dnnl_reorder_t* reorder, dnnl_stream_t stream)
{
    if (reorder->need_reorder) {
        dnnl_exec_arg_t args[] = {
            { DNNL_ARG_FROM, reorder->src_mem },
            { DNNL_ARG_TO, reorder->dst_mem}
        };
        CHECK_DNNL(dnnl_primitive_execute(reorder->primitive, stream, 2, args));
    }

    return 0;
dnnl_error:
    return 1;
}

dnnl_status_t dnnl_reorder_destroy(dnnl_reorder_t* reorder)
{
    if (reorder->need_reorder) {
        CHECK_DNNL(dnnl_primitive_destroy(reorder->primitive));
        CHECK_DNNL(dnnl_memory_destroy(reorder->dst_mem));
    }
    return 0;
dnnl_error:
    return 1;
}