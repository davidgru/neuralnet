#pragma once


#if defined(AI_LOG_LEVEL) && (AI_LOG_LEVEL >= 4)
#define LOG_TRACE(fmt, ...) log_trace(fmt __VA_OPT__(,) __VA_ARGS__)
void log_trace(const char* fmt, ...);
#else
#define LOG_TRACE(fmt, ...)
#endif

#if defined(AI_LOG_LEVEL) && (AI_LOG_LEVEL >= 3)
#define LOG_INFO(fmt, ...) log_info(fmt __VA_OPT__(,) __VA_ARGS__)
void log_info(const char* fmt, ...);
#else
#define LOG_INFO(fmt, ...)
#endif

#if defined(AI_LOG_LEVEL) && (AI_LOG_LEVEL >= 2)
#define LOG_WARN(fmt, ...) log_warn(fmt __VA_OPT__(,) __VA_ARGS__)
void log_warn(const char* fmt, ...);
#else
#define LOG_WARN(fmt, ...)
#endif

#if defined(AI_LOG_LEVEL) && (AI_LOG_LEVEL >= 1)
#define LOG_ERROR(fmt, ...) log_error(fmt __VA_OPT__(,) __VA_ARGS__)
void log_error(const char* fmt, ...);
#else
#define LOG_ERROR(fmt, ...)
#endif

