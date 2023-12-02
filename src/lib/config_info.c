#include <stdio.h>


#include "config_info.h"


#define XSTR(s) STR(s)
#define STR(s) #s


void dump_compile_time_config()
{

    printf("Library was compiled with the following configuration:\n");

#if defined(LOG_LEVEL)
    printf("    LOG_LEVEL=" XSTR(LOG_LEVEL) "\n");
#endif

#if defined(DEBUG)
    printf("    DEBUG=" XSTR(DEBUG) "\n");
#endif

#if defined(USE_AVX)
    printf("    USE_AVX=" XSTR(USE_AVX) "\n");
#endif

};