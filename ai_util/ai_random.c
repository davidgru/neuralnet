
#include "ai_random.h"

#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>

#include <time.h>
#include <math.h>

#define PI 3.14f

static bool inited = false;

static float random_uniform()
{
    return ((float)rand() + 1) / (float)(RAND_MAX + 2);
}

static float init_random()
{
    srand(time(0));
    inited = true;
}


float AI_RandomNormal(float mean, float stddev)
{
    if (!inited)
        init_random();

    float v1 = random_uniform();
    float v2 = random_uniform();

    return sqrtf(-2.0f * logf(v1)) * cosf(2.0f * PI * v2) * stddev + mean;
}

float AI_RandomUniform(float min, float max)
{
    if (!inited)
        init_random();

    return random_uniform() * (max - min) + min;
}
