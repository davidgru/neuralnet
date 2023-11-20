#include <stdio.h>


#include "config_info.h"


#define XSTR(s) STR(s)
#define STR(s) #s


void dump_compile_time_config()
{

    printf("Library was compiled with the following configuration:\n");

#if defined(AI_LOG_LEVEL)
    printf("    AI_LOG_LEVEL=" XSTR(AI_LOG_LEVEL) "\n");
#endif

#if defined(AI_DEBUG)
    printf("    AI_DEBUG=" XSTR(AI_DEBUG) "\n");
#endif

#if defined(AI_USE_AVX)
    printf("    AI_USE_AVX=" XSTR(AI_USE_AVX) "\n");
#endif

};