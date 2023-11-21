
/* Based on: http://cs231n.stanford.edu/handouts/linear-backprop.pdf */


#include "ai_linear_layer.h"


#include <malloc.h>
#include <string.h>
#if defined(AI_USE_AVX)
#include <immintrin.h>
#endif

#include "util/ai_math.h"

#include "log.h"


#define LINEAR_WEIGHTS_OUTPUT_DIM 2
#define LINEAR_WEIGHTS_INPUT_DIM  3



// A fully connected layer
typedef struct linear_layer_t {
    AI_Layer hdr;
    tensor_t weights;
    tensor_t bias;
    tensor_t d_weights;
    tensor_t d_bias;
    float learning_rate;
    uint32_t dummy;
} linear_layer_t;


static void matrix_product(float* m1, float* m2, float* output, size_t height1, size_t width2, size_t sharedDim);
static void matrix_product_t1(float* m1, float* m2, float* output, size_t height1, size_t width2, size_t sharedDim);
static void matrix_product_t2(float* m1, float* m2, float* output, size_t height1, size_t width2, size_t sharedDim);

// Forward propagation through the layer
static void linear_layer_forward(AI_Layer* layer);
// Backward propagation through the layer
static void linear_layer_backward(AI_Layer* layer);
static void linear_layer_info(AI_Layer* layer);
// Deinit a linear layer
static void linear_layer_deinit(AI_Layer* layer);

// Init a linear layer
uint32_t linear_layer_init(AI_Layer** layer, void* create_info, AI_Layer* prev_layer)
{
    AI_LinearLayerCreateInfo* linear_create_info = (AI_LinearLayerCreateInfo*)create_info;

    /* TODO: sanity check that input is one-dimensional */

    *layer = (AI_Layer*)malloc(sizeof(linear_layer_t));
    if (*layer == NULL) {
        return 1;
    }

    linear_layer_t* linear_layer = (linear_layer_t*)*layer;


    /* fill header information*/
    linear_layer->hdr.input_shape = prev_layer->output_shape;
    linear_layer->hdr.output_shape.dims[TENSOR_BATCH_DIM]
        = linear_layer->hdr.input_shape.dims[TENSOR_BATCH_DIM];
    linear_layer->hdr.output_shape.dims[TENSOR_CHANNEL_DIM] = linear_create_info->output_size;
    linear_layer->hdr.output_shape.dims[TENSOR_HEIGHT_DIM] = 1;
    linear_layer->hdr.output_shape.dims[TENSOR_WIDTH_DIM] = 1;

    /* virtual functions */
    linear_layer->hdr.forward = linear_layer_forward;
    linear_layer->hdr.backward = linear_layer_backward;
    linear_layer->hdr.info = linear_layer_info;
    linear_layer->hdr.deinit = linear_layer_deinit;

    /* allocate owned memory */
    tensor_allocate(&linear_layer->hdr.output, &linear_layer->hdr.output_shape);
    tensor_allocate(&linear_layer->hdr.gradient, &linear_layer->hdr.input_shape);

    /* For now implicitly flatten input. Might be benefical to implement an flatten layer in
        future. */
    tensor_shape_t weights_shape = {
        .dims[0] = 0,
        .dims[1] = 0,
        .dims[LINEAR_WEIGHTS_OUTPUT_DIM] = linear_create_info->output_size,
        .dims[LINEAR_WEIGHTS_INPUT_DIM] = prev_layer->output_shape.dims[TENSOR_CHANNEL_DIM] 
            * prev_layer->output_shape.dims[TENSOR_HEIGHT_DIM]
            * prev_layer->output_shape.dims[TENSOR_WIDTH_DIM]
    };
    tensor_allocate(&linear_layer->weights, &weights_shape);
    tensor_allocate(&linear_layer->d_weights, &weights_shape);

    tensor_shape_t bias_shape = {
        .dims[0] = 0,
        .dims[1] = 0,
        .dims[2] = 0,
        .dims[3] = linear_create_info->output_size
    };
    tensor_allocate(&linear_layer->bias, &bias_shape);
    tensor_allocate(&linear_layer->d_bias, &bias_shape);

    linear_layer->learning_rate = linear_create_info->learning_rate;

    /* Initialise weights and bias */
    
    float* weights_data = tensor_get_data(&linear_layer->weights);
    const size_t weights_size = tensor_size_from_shape(&weights_shape);
    for (size_t i = 0; i < weights_size; i++) {
        weights_data[i] = linear_create_info->weight_init(
            weights_shape.dims[LINEAR_WEIGHTS_INPUT_DIM],
            weights_shape.dims[LINEAR_WEIGHTS_OUTPUT_DIM]
        );
    }

    float* bias_data = tensor_get_data(&linear_layer->bias);
    const size_t bias_size = tensor_size_from_shape(&bias_shape);
    for (size_t i = 0; i < bias_size; i++) {
        bias_data[i] = linear_create_info->bias_init(
            weights_shape.dims[LINEAR_WEIGHTS_INPUT_DIM],
            weights_shape.dims[LINEAR_WEIGHTS_OUTPUT_DIM]
        );
    }

    return 0;
}


static void linear_layer_forward(AI_Layer* layer)
{
    linear_layer_t* linear_layer = (linear_layer_t*)layer;


    const tensor_shape_t* input_shape = tensor_get_shape(linear_layer->hdr.input);
    const tensor_shape_t* output_shape = tensor_get_shape(&linear_layer->hdr.output);


    float* input = tensor_get_data(linear_layer->hdr.input);
    float* output = tensor_get_data(&linear_layer->hdr.output);
    float* weights = tensor_get_data(&linear_layer->weights);
    float* bias = tensor_get_data(&linear_layer->bias);


    const size_t batch_size = input_shape->dims[TENSOR_BATCH_DIM];
    const size_t per_batch_input_size = tensor_size_from_shape(input_shape) / batch_size;
    const size_t output_size = output_shape->dims[TENSOR_CHANNEL_DIM];


    /* output = input * weights */
    matrix_product(input, weights, output, batch_size, output_size, per_batch_input_size);

    /* output += bias */
    for (size_t i = 0; i < batch_size; i++) {
        float* batch_output = output + i * output_size;
        AI_VectorAdd(batch_output, bias, output_size);
    }
}

static void linear_layer_backward(AI_Layer* layer)
{
    linear_layer_t* linear_layer = (linear_layer_t*)layer;


    const tensor_shape_t* input_shape = tensor_get_shape(linear_layer->hdr.input);
    const tensor_shape_t* output_shape = tensor_get_shape(&linear_layer->hdr.output);
    const tensor_shape_t* weights_shape = tensor_get_shape(&linear_layer->weights);


    float* input = tensor_get_data(linear_layer->hdr.input);
    float* output = tensor_get_data(&linear_layer->hdr.output);
    float* weights = tensor_get_data(&linear_layer->weights);
    float* bias = tensor_get_data(&linear_layer->bias);
    float* d_weights = tensor_get_data(&linear_layer->d_weights);
    float* gradient = tensor_get_data(&linear_layer->hdr.gradient);
    float* prev_gradient = tensor_get_data(linear_layer->hdr.prev_gradient);


    const size_t batch_size = input_shape->dims[TENSOR_BATCH_DIM];
    const size_t per_batch_input_size = tensor_size_from_shape(input_shape) / batch_size;
    const size_t output_channels = output_shape->dims[TENSOR_CHANNEL_DIM];
    const size_t weights_size = tensor_size_from_shape(weights_shape);

    /* Calculate gradient for backprop: gradient = prev_gradient * weights.T */
    matrix_product_t2(prev_gradient, weights, gradient, batch_size, per_batch_input_size,
        output_channels);

    /* Calculate gradient of weights: d_weights = weights.T * prev_gradient */
    matrix_product_t1(input, prev_gradient, d_weights, per_batch_input_size, output_channels,
        batch_size);
    /* d_weights *= learning_rate / batch_size. TODO: combine scale and subtraction */
    AI_VectorScale(d_weights, linear_layer->learning_rate * (1.0f / batch_size), weights_size);
    /* Gradient step: weights -= d_weights */
    AI_VectorSub(weights, d_weights, weights_size);

    /* Adjust bias (use layer->dw buffer) */
    memset(d_weights, 0, output_channels * sizeof(float));
    for (size_t i = 0; i < batch_size; i++) {
        AI_VectorAdd(d_weights, prev_gradient + i * output_channels, output_channels);
    }
    AI_VectorScale(d_weights, linear_layer->learning_rate * (1.0f / batch_size), output_channels);
    AI_VectorSub(bias, d_weights, output_channels);
}


static void linear_layer_info(AI_Layer* layer)
{
    // linear_layer_t* _layer = (linear_layer_t*)layer;
    // const size_t input_size = _layer->hdr.input_width;
    // const size_t output_size = _layer->hdr.output_width;
    // const size_t weight_size = input_size * output_size;

    // LOG_INFO("linear layer info: in: %d out: %d wdist: (%f +/- %f) bdist (%f +/- %f)\n",
    //     (int)input_size, (int)output_size, AI_Mean(_layer->w, weight_size),
    //     AI_Stddev(_layer->w, weight_size), AI_Mean(_layer->b, output_size),
    //     AI_Stddev(_layer->b, output_size));
}

static void linear_layer_deinit(AI_Layer* layer)
{
    linear_layer_t* linear_layer = (linear_layer_t*)layer;
    if (linear_layer != NULL) {
        tensor_destory(&linear_layer->hdr.output);
        tensor_destory(&linear_layer->hdr.gradient);
        tensor_destory(&linear_layer->weights);
        tensor_destory(&linear_layer->d_weights);
        tensor_destory(&linear_layer->bias);
        tensor_destory(&linear_layer->d_bias);
        free(linear_layer);
    }
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
                sum += m1[r * sharedDim + s] * m2[s * width2 + c];
            }
            output[r * owidth + c] = sum;
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
            output[r * owidth + c] = sum;
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
            output[r * owidth + c] = sum;
        }
    }
}


#endif
