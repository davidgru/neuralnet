/* Based on: http://cs231n.stanford.edu/handouts/linear-backprop.pdf */
#include <malloc.h>
#include <string.h>
#if defined(USE_AVX)
#include <immintrin.h>
#endif

#include "util/ai_math.h"
#include "tensor_impl.h"
#include "log.h"

#include "linear_layer.h"


#define NUM_LINEAR_LAYER_PARAMS 2
#define LINEAR_LAYER_WEIGHTS_PARAM 0
#define LINEAR_LAYER_BIAS_PARAM 1


#define LINEAR_WEIGHTS_OUTPUT_DIM 2
#define LINEAR_WEIGHTS_INPUT_DIM  3


typedef struct linear_layer_t {
    tensor_t weights;
    tensor_t bias;
    tensor_t d_weights;
    tensor_t d_bias;
    layer_param_ref_t param_refs[NUM_LINEAR_LAYER_PARAMS];
    uint32_t dummy;
} linear_layer_t;


static uint32_t linear_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
);

static uint32_t linear_layer_get_params(
    layer_context_t* context,
    layer_param_ref_list_t* out_layer_params
);

static uint32_t linear_layer_deinit(layer_context_t* context);

static uint32_t linear_layer_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t* out_output
);

static uint32_t linear_layer_backward(
    layer_context_t* context,
    const tensor_t* input,
    const tensor_t* output,
    const tensor_t* prev_gradient,
    tensor_t* out_gradient
);

static uint32_t linear_layer_calc_output_shape(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
);


const layer_impl_t linear_layer_impl = {
    .init_func = linear_layer_init,
    .get_param_func = linear_layer_get_params,
    .deinit_func = linear_layer_deinit,
    .forward_func = linear_layer_forward,
    .backward_func = linear_layer_backward,
    .calc_output_size = linear_layer_calc_output_shape,
    .layer_context_size = sizeof(linear_layer_t)
};


static void matrix_product(const float* m1, const float* m2, float* output, size_t height1, size_t width2, size_t sharedDim);
static void matrix_product_t1(const float* m1, const float* m2, float* output, size_t height1, size_t width2, size_t sharedDim);
static void matrix_product_t2(const float* m1, const float* m2, float* output, size_t height1, size_t width2, size_t sharedDim);


static uint32_t linear_layer_init(
    layer_context_t* context,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape,
    const tensor_shape_t* output_shape
)
{
    linear_layer_t* linear_layer = (linear_layer_t*)context;
    const linear_layer_create_info_t* linear_create_info = (linear_layer_create_info_t*)create_info;


    /* For now implicitly flatten input. Might be benefical to implement an flatten layer in
        future. */
    tensor_shape_t weights_shape = {
        .dims[0] = 0,
        .dims[1] = 0,
        .dims[LINEAR_WEIGHTS_OUTPUT_DIM] = output_shape->dims[TENSOR_CHANNEL_DIM],
        .dims[LINEAR_WEIGHTS_INPUT_DIM] = input_shape->dims[TENSOR_CHANNEL_DIM]
            * input_shape->dims[TENSOR_HEIGHT_DIM]
            * input_shape->dims[TENSOR_WIDTH_DIM]
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


    /* need to register the params for the optimizer */
    linear_layer->param_refs[LINEAR_LAYER_WEIGHTS_PARAM].param = &linear_layer->weights;
    linear_layer->param_refs[LINEAR_LAYER_WEIGHTS_PARAM].gradient = &linear_layer->d_weights;
    linear_layer->param_refs[LINEAR_LAYER_BIAS_PARAM].param = &linear_layer->bias;
    linear_layer->param_refs[LINEAR_LAYER_BIAS_PARAM].gradient = &linear_layer->d_bias;


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
};


static uint32_t linear_layer_get_params(
    layer_context_t* context,
    layer_param_ref_list_t* out_layer_params
)
{
    linear_layer_t* linear_layer = (linear_layer_t*)context;

    out_layer_params->param_refs = linear_layer->param_refs;
    out_layer_params->num_params = NUM_LINEAR_LAYER_PARAMS;
    return 0;
}


static uint32_t linear_layer_deinit(layer_context_t* context)
{
    linear_layer_t* linear_layer = (linear_layer_t*)context;
    
    tensor_destory(&linear_layer->weights);
    tensor_destory(&linear_layer->d_weights);
    tensor_destory(&linear_layer->bias);
    tensor_destory(&linear_layer->d_bias);
}


static uint32_t linear_layer_forward(
    layer_context_t* context,
    layer_forward_kind_t forward_kind,
    const tensor_t* input,
    tensor_t* out_output
)
{
    linear_layer_t* linear_layer = (linear_layer_t*)context;


    const tensor_shape_t* input_shape = tensor_get_shape(input);
    const tensor_shape_t* output_shape = tensor_get_shape(out_output);


    const float* input_data = tensor_get_data_const(input);
    const float* weights = tensor_get_data_const(&linear_layer->weights);
    const float* bias = tensor_get_data_const(&linear_layer->bias);
    float* output_data = tensor_get_data(out_output);


    const size_t batch_size = input_shape->dims[TENSOR_BATCH_DIM];
    const size_t per_batch_input_size = tensor_size_from_shape(input_shape) / batch_size;
    const size_t output_size = output_shape->dims[TENSOR_CHANNEL_DIM];


    /* output = input * weights */
    matrix_product(input_data, weights, output_data, batch_size, output_size, per_batch_input_size);

    /* output += bias */
    for (size_t i = 0; i < batch_size; i++) {
        float* batch_output = output_data + i * output_size;
        VectorAdd(batch_output, bias, output_size);
    }
}


static uint32_t linear_layer_backward(
    layer_context_t* context,
    const tensor_t* input,
    const tensor_t* output,
    const tensor_t* prev_gradient,
    tensor_t* out_gradient
)
{
    linear_layer_t* linear_layer = (linear_layer_t*)context;


    const tensor_shape_t* input_shape = tensor_get_shape(input);
    const tensor_shape_t* output_shape = tensor_get_shape(output);
    const tensor_shape_t* weights_shape = tensor_get_shape(&linear_layer->weights);


    const float* input_data = tensor_get_data_const(input);
    const float* prev_gradient_data = tensor_get_data_const(prev_gradient);
    float* gradient_data = tensor_get_data(out_gradient);
    float* weights = tensor_get_data(&linear_layer->weights);
    float* bias = tensor_get_data(&linear_layer->bias);
    float* d_weights = tensor_get_data(&linear_layer->d_weights);
    float* d_bias = tensor_get_data(&linear_layer->d_bias);


    const size_t batch_size = input_shape->dims[TENSOR_BATCH_DIM];
    const size_t per_batch_input_size = tensor_size_from_shape(input_shape) / batch_size;
    const size_t output_channels = output_shape->dims[TENSOR_CHANNEL_DIM];
    const size_t weights_size = tensor_size_from_shape(weights_shape);

    /* Calculate gradient for backprop: gradient = prev_gradient * weights.T */
    matrix_product_t2(prev_gradient_data, weights, gradient_data, batch_size, per_batch_input_size,
        output_channels);

    /* Calculate gradient of weights: d_weights = weights.T * prev_gradient */
    matrix_product_t1(input_data, prev_gradient_data, d_weights, per_batch_input_size, output_channels,
        batch_size);
    /* d_weights /= batch_size */
    VectorScale(d_weights, (1.0f / batch_size), weights_size);
    
    /* Calculate gradient of bias */
    memset(d_bias, 0, output_channels * sizeof(float));
    for (size_t i = 0; i < batch_size; i++) {
        VectorAdd(d_bias, prev_gradient_data + i * output_channels, output_channels);
    }
    VectorScale(d_bias, (1.0f / batch_size), output_channels);
}


static uint32_t linear_layer_calc_output_shape(
    tensor_shape_t* out_output_shape,
    const layer_create_info_t* create_info,
    const tensor_shape_t* input_shape
)
{
    linear_layer_create_info_t* linear_create_info = (linear_layer_create_info_t*)create_info;

    /* For now implicitly flatten input. Might be benefical to implement an flatten layer in
        future. */
    out_output_shape->dims[TENSOR_BATCH_DIM] = input_shape->dims[TENSOR_BATCH_DIM];
    out_output_shape->dims[TENSOR_CHANNEL_DIM] = linear_create_info->output_size;
    out_output_shape->dims[TENSOR_HEIGHT_DIM] = 1;
    out_output_shape->dims[TENSOR_WIDTH_DIM] = 1;    

    return 0;
}


#if defined(USE_AVX)


// AVX accelerated matrix product: output = m1 * m2
static void matrix_product(const float* m1, const float* m2, float* output, size_t height1, size_t width2, size_t sharedDim)
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
static void matrix_product_t1(const float* m1, const float* m2, float* output, size_t width1, size_t width2, size_t sharedDim)
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
static void matrix_product_t2(const float* m1, const float* m2, float* output, size_t height1, size_t height2, size_t sharedDim)
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


static void matrix_product(const float* m1, const float* m2, float* output, size_t height1, size_t width2, size_t sharedDim)
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

static void matrix_product_t1(const float* m1, const float* m2, float* output, size_t width1, size_t width2, size_t sharedDim)
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

static void matrix_product_t2(const float* m1, const float* m2, float* output, size_t height1, size_t height2, size_t sharedDim)
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
