
#include "ai_gradient_clipping.h"

#include <math.h>

void AI_ClipGradient(float* v, size_t size, float threshold)
{
    for (size_t i = 0; i < size; i++) {
        if (fabsf(v[i]) > threshold)
            v[i] = threshold * (v[i] / fabsf(v[i]));
    }
}
