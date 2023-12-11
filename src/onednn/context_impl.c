#include "log.h"

#include "context_impl.h"


static dnnl_engine_t engine = NULL;
static dnnl_stream_t stream = NULL;


uint32_t backend_context_init()
{
    dnnl_status_t status;

    status = dnnl_engine_create(&engine, dnnl_cpu, 0);
    if (status != dnnl_success) {
        LOG_ERROR("Failed to create oneDNN engine: %d\n", status);
        return 1;
    }

    status = dnnl_stream_create(&stream, engine, dnnl_stream_default_flags);
    if (status != dnnl_success) {
        dnnl_engine_destroy(engine);
        LOG_ERROR("Failed to create oneDNN stream: %d\n", status);
        return 1;
    }

    return 0;
}


dnnl_engine_t get_dnnl_engine()
{
    return engine;
}


dnnl_stream_t get_dnnl_stream()
{
    return stream;
}
