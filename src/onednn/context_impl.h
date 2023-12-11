#pragma once

#include "dnnl.h"

#include "context.h"


dnnl_engine_t get_dnnl_engine();
dnnl_stream_t get_dnnl_stream();
