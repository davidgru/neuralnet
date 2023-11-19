#pragma once

#include <stdio.h>

#include "dnnl.h"

#define CHECK_DNNL(call)\
    {\
        dnnl_status_t status = call;\
        if (status) {\
            printf("dnnl_error: %s %d\n", #call, status);\
            goto dnnl_error;\
        }\
    }
