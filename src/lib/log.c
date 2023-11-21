
#if defined(AI_LOG_LEVEL) && (AI_LOG_LEVEL >= 1)


#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "log.h"


static void log_internal(const char* prefix, const char* fmt, va_list args);


#if (AI_LOG_LEVEL >= 4)
void log_trace(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    log_internal("trace: ", fmt, args);
    va_end(args);
}
#endif

#if (AI_LOG_LEVEL >= 3)
void log_info(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    log_internal("info: ", fmt, args);
    va_end(args);
}
#endif

#if (AI_LOG_LEVEL >= 2)
void log_warn(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    log_internal("warn: ", fmt, args);
    va_end(args);
}
#endif

#if (AI_LOG_LEVEL >= 1)
void log_error(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    log_internal("error: ", fmt, args);
    va_end(args);
}
#endif


static void log_internal(const char* prefix, const char* fmt, va_list args)
{
    size_t prefixlen = strlen(prefix);
    size_t fmtlen = strlen(fmt);

    char* prefixed_fmt = (char*)malloc(prefixlen + fmtlen + 1);
    strcpy(prefixed_fmt, prefix);
    strcpy(prefixed_fmt + prefixlen, fmt);

    int ret = vprintf(prefixed_fmt, args);

    free(prefixed_fmt);
}


#endif
