
#include "ai_linear_layer.h"


#include <malloc.h>
#include <string.h>
// AVX intrinsic header
#include <immintrin.h>

#include "util/ai_math.h"
#include "util/ai_gradient_clipping.h"

#include "log.h"

// Based on: http://cs231n.stanford.edu/handouts/linear-backprop.pdf

/*
 * A fully connected layer

typedef struct AI_FCLayer {
    size_t input_size;
    size_t output_size;
    float* w; // weight matrix:         stored in layer             (input_size x output_size)
    float* b; // bias matrix:           stored in layer             (1 x output_size)
    float* x; // input matrix:          stored in previous layer    (1 x input_size)
    float* y; // output matrix:         stored in layer             (1 x output_size)
    float* dw; // weight gradients:     stored in layer             (input_size x output_size)
    float* dx; // input gradients:      stored in layer             (1 x input_size)
    float* dy; // output gradients:     stored in next layer        (1 x output_size)
} AI_FCLayer;

*/

// A fully connected layer
typedef struct linear_layer_t {
    AI_Layer hdr;
    float* w;
    float* b;
    float* dw;
    float learning_rate;
    float gradient_clipping_threshold;
    uint32_t dummy;
} linear_layer_t;


static void matrix_product(float* m1, float* m2, float* output, size_t height1, size_t width2, size_t sharedDim);
static void matrix_product_t1(float* m1, float* m2, float* output, size_t height1, size_t width2, size_t sharedDim);
static void matrix_product_t2(float* m1, float* m2, float* output, size_t height1, size_t width2, size_t sharedDim);

// Forward propagation through the layer
static void linear_layer_forward(AI_Layer* layer);
// Backward propagation through the layer
static void linear_layer_backward(AI_Layer* layer);
// Deinit a linear layer
static void linear_layer_deinit(AI_Layer* layer);

// Init a linear layer
uint32_t linear_layer_init(AI_Layer** layer, void* create_info, AI_Layer* prev_layer)
{
    AI_LinearLayerCreateInfo* _create_info = (AI_LinearLayerCreateInfo*)create_info;

    const size_t input_size = prev_layer->output_width * prev_layer->output_height * prev_layer->output_channels;
    const size_t output_size = _create_info->output_size;
    const size_t batch_size = prev_layer->mini_batch_size;
    const size_t weight_size = input_size * output_size;

    // Allocate memory for the layer
    size_t size = sizeof(linear_layer_t) + (weight_size + output_size + output_size * batch_size + input_size * batch_size + weight_size) * sizeof(float);
    *layer = (AI_Layer*)malloc(size);

    linear_layer_t* _layer = (linear_layer_t*)*layer;

    // Set layer attributes
    _layer->hdr.input_width = input_size;
    _layer->hdr.input_height = 1;
    _layer->hdr.input_channels = 1;
    _layer->hdr.output_width = output_size;
    _layer->hdr.output_height = 1;
    _layer->hdr.output_channels = 1;
    _layer->hdr.mini_batch_size = batch_size;
    _layer->hdr.forward = linear_layer_forward;
    _layer->hdr.backward = linear_layer_backward;
    _layer->hdr.deinit = linear_layer_deinit;

    _layer->learning_rate = _create_info->learning_rate;
    _layer->gradient_clipping_threshold = _create_info->gradient_clipping_threshold;

    // Assign buffers
    _layer->w = (float*)(_layer + 1);
    _layer->b = _layer->w + weight_size;
    _layer->hdr.output = _layer->b + output_size;
    _layer->hdr.gradient = _layer->hdr.output + output_size * batch_size;
    _layer->dw = _layer->hdr.gradient + input_size * batch_size;

    // Set those in the linking function
    _layer->hdr.input = 0;
    _layer->hdr.prev_gradient = 0;

    // Initialise weights and bias
    for (size_t i = 0; i < weight_size; i++)
        _layer->w[i] = _create_info->weight_init(input_size, output_size);
    for (size_t i = 0; i < output_size; i++)
        _layer->b[i] = _create_info->bias_init(input_size, output_size);

    LOG_INFO("linear layer initialized with: %d out: %d wdist: (%f +/- %f) bdist (%f +/- %f)\n",
        (int)input_size, (int)output_size, AI_Mean(_layer->w, weight_size),
        AI_Stddev(_layer->w, weight_size), AI_Mean(_layer->b, _layer->hdr.output_width),
        AI_Stddev(_layer->b, _layer->hdr.output_width));
}


static void linear_layer_forward(AI_Layer* layer)
{
    linear_layer_t* _layer = (linear_layer_t*)layer;

    matrix_product(_layer->hdr.input, _layer->w, _layer->hdr.output, _layer->hdr.mini_batch_size, _layer->hdr.output_width, _layer->hdr.input_width);
    
    // Add bias
    for (size_t i = 0; i < _layer->hdr.mini_batch_size; i++)
        AI_VectorAddAVX(_layer->hdr.output + i * _layer->hdr.output_width, _layer->b, _layer->hdr.output_width);
}

static void linear_layer_backward(AI_Layer* layer)
{
    linear_layer_t* _layer = (linear_layer_t*)layer;

    const size_t input_size = _layer->hdr.input_width;
    const size_t output_size = _layer->hdr.output_width;
    const size_t weights_size = input_size * _layer->hdr.output_width;
    const size_t mini_batch_size = _layer->hdr.mini_batch_size;
    const float learning_rate = _layer->learning_rate;

    float* w = _layer->w;
    float* b = _layer->b;
    float* y = _layer->hdr.output;
    float* x = _layer->hdr.input;
    float* dx = _layer->hdr.gradient;
    float* dy = _layer->hdr.prev_gradient;
    float* dw = _layer->dw;

    // Calculate gradients with respect to input
    matrix_product_t2(dy, w, dx, mini_batch_size, input_size, output_size);

    // Adjust weights
    matrix_product_t1(x, dy, dw, input_size, output_size, mini_batch_size); // Perform dw = x_t * dy
    AI_VectorScaleAVX(dw, learning_rate, weights_size);
    AI_ClipGradient(dw, weights_size, _layer->gradient_clipping_threshold);
    AI_VectorSubAVX(w, dw, weights_size); // Subtract dw from w

    // Adjust bias (use layer->dw buffer)
    memset(dw, 0, output_size * sizeof(float));
    for (size_t i = 0; i < mini_batch_size; i++)
        AI_VectorCopy(dw, dy + i * output_size, output_size);
    AI_VectorScaleAVX(dw, learning_rate, output_size);
    AI_ClipGradient(dw, output_size, _layer->gradient_clipping_threshold);
    AI_VectorSubAVX(b, dw, output_size);
}

static void linear_layer_deinit(AI_Layer* layer)
{
    linear_layer_t* _layer = (linear_layer_t*)layer;
    if (_layer)
        free(_layer);
}


#if defined(AI_USE_AVX)


// AVX accelerated matrix product: output = m1 * m2
static void matrix_product(float* m1, float* m2, float* output, size_t height1, size_t width2, size_t sharedDim)
{
    const size_t owidth = width2;
    const size_t oheight = height1;

    size_t c_unroll = owidth / 16 * 16;

    for (size_t r = 0; r < oheight; r++) {
        size_t c = 0;
        for (; c < c_unroll; c += 16) {
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            for (size_t s = 0; s < sharedDim; s++) {
                __m256 bc_m1 = _mm256_set1_ps(m1[r * sharedDim + s]);
                __m256 v1_m2 = _mm256_load_ps(m2 + s * width2 + c);
                __m256 v2_m2 = _mm256_load_ps(m2 + s * width2 + c + 8);
                __m256 prod1 = _mm256_mul_ps(bc_m1, v1_m2);
                __m256 prod2 = _mm256_mul_ps(bc_m1, v2_m2);
                sum1 = _mm256_add_ps(sum1, prod1);
                sum2 = _mm256_add_ps(sum2, prod2);
            }
            _mm256_storeu_ps(output + r * owidth + c, sum1);
            _mm256_storeu_ps(output + r * owidth + c + 8, sum2);
        }
        for (; c < owidth; c++) {
            float sum = 0.0f;
            for (size_t s = 0; s < sharedDim; s++)
                sum += m1[r * sharedDim + s] * m2[s * width2 + c];
            output[r * owidth + c] = sum;
        }
    }
}

// AVX accelerated matrix product, where m1 is transposed: output = m1_t * m2
static void matrix_product_t1(float* m1, float* m2, float* output, size_t width1, size_t width2, size_t sharedDim)
{
    const size_t owidth = width2;
    const size_t oheight = width1;

    size_t c_unroll = owidth / 16 * 16;

    for (size_t r = 0; r < oheight; r++) {
        size_t c = 0;
        for (; c < c_unroll; c += 16) {
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            for (size_t s = 0; s < sharedDim; s++) {
                __m256 bc_m1 = _mm256_set1_ps(m1[s * width1 + r]);
                __m256 v1_m2 = _mm256_load_ps(m2 + s * width2 + c);
                __m256 v2_m2 = _mm256_load_ps(m2 + s * width2 + c + 8);
                __m256 prod1 = _mm256_mul_ps(bc_m1, v1_m2);
                __m256 prod2 = _mm256_mul_ps(bc_m1, v2_m2);
                sum1 = _mm256_add_ps(sum1, prod1);
                sum2 = _mm256_add_ps(sum2, prod2);
            }
            _mm256_storeu_ps(output + r * owidth + c, sum1);
            _mm256_storeu_ps(output + r * owidth + c + 8, sum2);
        }
        for (; c < owidth; c++) {
            float sum = 0.0f;
            for (size_t s = 0; s < sharedDim; s++)
                sum += m1[s * width1 + r] * m2[s * width2 + c];
            output[r * owidth + c] = sum;
        }
    }
}

// AVX accelerated matrix product, where m2 is transposed: output = m1 * m2_t
static void matrix_product_t2(float* m1, float* m2, float* output, size_t height1, size_t height2, size_t sharedDim)
{
    const size_t owidth = height2;
    const size_t oheight = height1;

    size_t s_unroll = sharedDim / 8 * 8;

    for (size_t r = 0; r < oheight; r++) {
        for (size_t c = 0; c < owidth; c++) {
            __m256 vsum = _mm256_setzero_ps();
            size_t s = 0;
            for (; s < s_unroll; s += 8) {
                __m256 v1 = _mm256_loadu_ps(m1 + r * sharedDim + s);
                __m256 v2 = _mm256_loadu_ps(m2 + c * sharedDim + s);
                vsum = _mm256_add_ps(vsum, _mm256_mul_ps(v1, v2));
            }
            // Horizontal add
            __m256 t1 = _mm256_hadd_ps(vsum, vsum);
            __m256 t2 = _mm256_hadd_ps(t1, t1);
            __m128 t3 = _mm256_extractf128_ps(t2, 1);
            __m128 t4 = _mm_add_ss(_mm256_castps256_ps128(t2), t3);
            float sum = _mm_cvtss_f32(t4);

            for (; s < sharedDim; s++)
                sum += m1[r * sharedDim + s] * m2[c * sharedDim + s];
            output[r * owidth + c] = sum;
        }
    }
}


#else


static void matrix_product(float* m1, float* m2, float* output, size_t height1, size_t width2, size_t sharedDim)
{
    const size_t owidth = width2;
    const size_t oheight = height1;

    for (size_t r = 0; r < oheight; r++) {
        for (size_t c = 0; c < owidth; c++) {
            float sum = 0.0f;
            for (size_t s = 0; s < sharedDim; s++) {
                float a1 = m1[r * height1 + s];
                float a2 = m2[s * width2 + c];
                sum += a1 * a2;
            }
            output[r + owidth + c] = sum;
        }
    }
}

static void matrix_product_t1(float* m1, float* m2, float* output, size_t width1, size_t width2, size_t sharedDim)
{
    const size_t owidth = width2;
    const size_t oheight = width1;

    for (size_t r = 0; r < oheight; r++) {
        for (size_t c = 0; c < owidth; c++) {
            float sum = 0.0f;
            for (size_t s = 0; s < sharedDim; s++) {
                sum += m1[s * width1 + r] * m2[s * width2 + c];
            }
            output[r + owidth + c] = sum;
        }
    }
}

static void matrix_product_t2(float* m1, float* m2, float* output, size_t height1, size_t height2, size_t sharedDim)
{
    const size_t owidth = height2;
    const size_t oheight = height1;

    for (size_t r = 0; r < oheight; r++) {
        for (size_t c = 0; c < owidth; c++) {
            float sum = 0.0f;
            for (size_t s = 0; s < sharedDim; s++) {
                sum += m1[r * sharedDim + s] * m2[c * sharedDim + s];
            }
            output[r + owidth + c] = sum;
        }
    }
}


#endif
