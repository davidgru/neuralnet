
#include "ai_layer/ai_base_layer.h"
#include "ai_util/ai_loss.h"

#include "ai_net.h"

#include "ai_datasets/ai_mnist.h"

#include <stdio.h>
#include <string.h>

void train_callback(AI_TrainingProgress* p)
{
    printf("Epoch %3d | Train loss %f | Train accuracy %5.3f%% | Test loss %f | Test accuracy %5.3f%%\n",
        p->epoch,
        p->train_loss,
        p->train_accuracy,
        p->test_loss,
        p->test_accuracy
    );
}

void train_lenet1()
{
    size_t epochs = 10000;
    float learning_rate = 0.002f;
    float clipping_threshold = 1000.0f;

    AI_Net net;
    AI_MnistDataset mnist;

    AI_ConvolutionalLayerCreateInfo c1_info;
    c1_info.output_channels = 4;
    c1_info.filter_width = 5;
    c1_info.learning_rate = learning_rate;
    c1_info.gradient_clipping_threshold = clipping_threshold;
    c1_info.weight_init = AI_ConvWeightInitXavier;
    c1_info.bias_init = AI_ConvBiasInitZeros;

    AI_ActivationLayerCreateInfo a1_info;
    a1_info.activation_function = AI_ACTIVATION_FUNCTION_TANH;

    AI_PoolingLayerCreateInfo p1_info;
    p1_info.kernel_width = 2;
    p1_info.pooling_operation = AI_POOLING_AVERAGE;

    AI_ConvolutionalLayerCreateInfo c2_info;
    c2_info.output_channels = 12;
    c2_info.filter_width = 5;
    c2_info.learning_rate = learning_rate;
    c2_info.gradient_clipping_threshold = clipping_threshold;
    c2_info.weight_init = AI_ConvWeightInitXavier;
    c2_info.bias_init = AI_ConvBiasInitZeros;

    AI_ActivationLayerCreateInfo a2_info;
    a2_info.activation_function = AI_ACTIVATION_FUNCTION_TANH;

    AI_PoolingLayerCreateInfo p2_info;
    p2_info.kernel_width = 2;
    p2_info.pooling_operation = AI_POOLING_AVERAGE;

    AI_ConvolutionalLayerCreateInfo c3_info;
    c3_info.output_channels = 10;
    c3_info.filter_width = 4;
    c3_info.learning_rate = learning_rate;
    c3_info.gradient_clipping_threshold = clipping_threshold;
    c3_info.weight_init = AI_ConvWeightInitXavier;
    c3_info.bias_init = AI_ConvBiasInitZeros;

    AI_ActivationLayerCreateInfo a3_info;
    a3_info.activation_function = AI_ACTIVATION_FUNCTION_SIGMOID;

    AI_LayerCreateInfo create_infos[] = {
        { AI_CONVOLUTIONAL_LAYER, &c1_info },
        { AI_ACTIVATION_LAYER,    &a1_info },
        { AI_POOLING_LAYER,       &p1_info },
        { AI_CONVOLUTIONAL_LAYER, &c2_info },
        { AI_ACTIVATION_LAYER,    &a2_info },
        { AI_POOLING_LAYER,       &p2_info },
        { AI_CONVOLUTIONAL_LAYER, &c3_info },
        { AI_ACTIVATION_LAYER,    &a3_info },
    };

    AI_NetInit(&net, 28, 28, 1, 1, create_infos, 8);
    AI_MnistDatasetLoad(&mnist, "D:/dev/tools/datasets/mnist", 0);

    AI_NetTrain(&net, mnist.train_images, mnist.test_images, mnist.train_labels, mnist.test_labels, mnist.num_train_images, mnist.num_test_images, epochs, learning_rate, 1, train_callback);

    AI_NetDeinit(&net);
    AI_MnistDatasetFree(&mnist);
}


#include "ai_dnnl_model.h"


void dnnl_train_callback(ai_dnnl_training_progress_t* p)
{
    printf("Epoch %3d | Train loss %f | Train acc %5.3f%% | Test loss %f | Test acc %5.3f%%\n",
        p->epoch,
        p->train_loss,
        p->train_acc,
        p->test_loss,
        p->test_acc
    );

}

void test_dnnl_model()
{
    size_t num_epochs = 1000;
    size_t N = 1;
    float learning_rate = 0.05f;

    AI_MnistDataset mnist;
    ai_dnnl_model_t* model;

    ai_dnnl_linear_layer_create_info_t l1_info;
    l1_info.OC = 300;
    l1_info.learning_rate = learning_rate;
    l1_info.allow_reorder = 1;
    l1_info.weight_init = ai_dnnl_linear_layer_weight_init_kind_xavier;
    l1_info.bias_init = ai_dnnl_linear_layer_bias_init_kind_zeros;

    ai_dnnl_activation_layer_create_info_t a1_info;
    a1_info.activation = ai_dnnl_activation_kind_logistic;
    
    ai_dnnl_linear_layer_create_info_t l2_info;
    l2_info.OC = 10;
    l2_info.allow_reorder = 1;
    l2_info.learning_rate = learning_rate;
    l2_info.weight_init = ai_dnnl_linear_layer_weight_init_kind_xavier;
    l2_info.bias_init = ai_dnnl_linear_layer_bias_init_kind_zeros;
    
    ai_dnnl_activation_layer_create_info_t a2_info;
    a2_info.activation = ai_dnnl_activation_kind_logistic;

    ai_dnnl_layer_create_info_t create_infos[] = {
        { ai_dnnl_layer_kind_linear, &l1_info },
        { ai_dnnl_layer_kind_activation, &a1_info },
        { ai_dnnl_layer_kind_linear, &l2_info },
        { ai_dnnl_layer_kind_activation, &a2_info }
    };


    ai_input_dims_t input_shape;
    input_shape.N = N;
    input_shape.C = 1;
    input_shape.H = 28;
    input_shape.W = 28;
    
    ai_dnnl_model_create(&model, &input_shape, 4, create_infos, ai_dnnl_loss_mse);
    AI_MnistDatasetLoad(&mnist, "D:/dev/tools/datasets/mnist", 0);

    printf("Model created\n");

    uint32_t status = ai_dnnl_model_train(model, 60000, mnist.train_images, mnist.train_labels, 10000, mnist.test_images, mnist.test_labels, num_epochs, dnnl_train_callback);

    printf("status: %d\n", status);
}

void ai_2_layer_net()
{
    size_t num_epochs = 1000;
    size_t N = 1;
    float learning_rate = 0.1f;
    float clipping_threshold = 1000.0f;

    AI_MnistDataset mnist;
    AI_Net model;

    AI_LinearLayerCreateInfo l1_info;
    l1_info.output_size = 300;
    l1_info.learning_rate = learning_rate;
    l1_info.gradient_clipping_threshold = clipping_threshold;
    l1_info.weight_init = AI_LinearWeightInitXavier;
    l1_info.bias_init = AI_LinearBiasInitZeros;

    AI_ActivationLayerCreateInfo a1_info;
    a1_info.activation_function = AI_ACTIVATION_FUNCTION_SIGMOID;

    AI_LinearLayerCreateInfo l2_info;
    l2_info.output_size = 10;
    l2_info.learning_rate = learning_rate;
    l2_info.gradient_clipping_threshold = clipping_threshold;
    l2_info.weight_init = AI_LinearWeightInitXavier;
    l2_info.bias_init = AI_LinearBiasInitZeros;
    
    AI_ActivationLayerCreateInfo a2_info;
    a2_info.activation_function = AI_ACTIVATION_FUNCTION_SIGMOID;

    AI_LayerCreateInfo create_infos[] = {
        { AI_LINEAR_LAYER, &l1_info },
        { AI_ACTIVATION_LAYER, &a1_info },
        { AI_LINEAR_LAYER, &l2_info },
        { AI_ACTIVATION_LAYER, &a2_info }
    };
    
    AI_NetInit(&model, 28, 28, 1, N, create_infos, 4);
    AI_MnistDatasetLoad(&mnist, "D:/dev/tools/datasets/mnist", 0);

    printf("Model created\n");
    AI_NetTrain(&model, mnist.train_images, mnist.test_images, mnist.train_labels, mnist.test_labels, 60000, 10000, num_epochs, learning_rate, N, train_callback);
}

void train_lenet5()
{
    size_t epochs = 10000;
    float learning_rate = 0.1f;
    float clipping_threshold = 1000.0f;
    float dropout_rate = 0.5f;

    AI_Net net;
    AI_MnistDataset mnist;

    AI_ConvolutionalLayerCreateInfo c1_info;
    c1_info.output_channels = 6;
    c1_info.filter_width = 5;
    c1_info.learning_rate = learning_rate;
    c1_info.gradient_clipping_threshold = clipping_threshold;
    c1_info.weight_init = AI_ConvWeightInitXavier;
    c1_info.bias_init = AI_ConvBiasInitZeros;

    AI_ActivationLayerCreateInfo a1_info;
    a1_info.activation_function = AI_ACTIVATION_FUNCTION_TANH;

    AI_PoolingLayerCreateInfo p1_info;
    p1_info.kernel_width = 2;
    p1_info.pooling_operation = AI_POOLING_AVERAGE;

    AI_ConvolutionalLayerCreateInfo c2_info;
    c2_info.output_channels = 16;
    c2_info.filter_width = 5;
    c2_info.learning_rate = learning_rate;
    c2_info.gradient_clipping_threshold = clipping_threshold;
    c2_info.weight_init = AI_ConvWeightInitXavier;
    c2_info.bias_init = AI_ConvBiasInitZeros;

    AI_ActivationLayerCreateInfo a2_info;
    a2_info.activation_function = AI_ACTIVATION_FUNCTION_TANH;

    AI_PoolingLayerCreateInfo p2_info;
    p2_info.kernel_width = 2;
    p2_info.pooling_operation = AI_POOLING_AVERAGE;

    AI_LinearLayerCreateInfo l3_info;
    l3_info.output_size = 120;
    l3_info.learning_rate = learning_rate;
    l3_info.gradient_clipping_threshold = clipping_threshold;
    l3_info.weight_init = AI_LinearWeightInitXavier;
    l3_info.bias_init = AI_LinearBiasInitZeros;

    AI_ActivationLayerCreateInfo a3_info;
    a3_info.activation_function = AI_ACTIVATION_FUNCTION_TANH;

    AI_DropoutLayerCreateInfo d3_info;
    d3_info.dropout_rate = dropout_rate;

    AI_LinearLayerCreateInfo l4_info;
    l4_info.output_size = 84;
    l4_info.learning_rate = learning_rate;
    l4_info.gradient_clipping_threshold = clipping_threshold;
    l4_info.weight_init = AI_LinearWeightInitXavier;
    l4_info.bias_init = AI_LinearBiasInitZeros;

    AI_ActivationLayerCreateInfo a4_info;
    a4_info.activation_function = AI_ACTIVATION_FUNCTION_TANH;
    
    AI_DropoutLayerCreateInfo d4_info;
    d4_info.dropout_rate = dropout_rate;

    AI_LinearLayerCreateInfo l5_info;
    l5_info.output_size = 10;
    l5_info.learning_rate = learning_rate;
    l5_info.gradient_clipping_threshold = clipping_threshold;
    l5_info.weight_init = AI_LinearWeightInitXavier;
    l5_info.bias_init = AI_LinearBiasInitZeros;

    AI_ActivationLayerCreateInfo a5_info;
    a5_info.activation_function = AI_ACTIVATION_FUNCTION_TANH;


    AI_LayerCreateInfo create_infos[] = {
        { AI_CONVOLUTIONAL_LAYER, &c1_info },
        { AI_ACTIVATION_LAYER,    &a1_info },
        { AI_POOLING_LAYER,       &p1_info },
        { AI_CONVOLUTIONAL_LAYER, &c2_info },
        { AI_ACTIVATION_LAYER,    &a2_info },
        { AI_POOLING_LAYER,       &p2_info },
        { AI_LINEAR_LAYER,        &l3_info },
        { AI_ACTIVATION_LAYER,    &a3_info },
        { AI_LINEAR_LAYER,        &l4_info },
        { AI_ACTIVATION_LAYER,    &a4_info },
        { AI_LINEAR_LAYER,        &l5_info },
        { AI_ACTIVATION_LAYER,    &a5_info },
    };

    AI_MnistDatasetLoad(&mnist, "D:/dev/tools/datasets/mnist", 2);
    AI_NetInit(&net, 32, 32, 1, 1, create_infos, 12);

    AI_NetTrain(&net, mnist.train_images, mnist.test_images, mnist.train_labels, mnist.test_labels, mnist.num_train_images, mnist.num_test_images, epochs, learning_rate, 1, train_callback);

    AI_NetDeinit(&net);
    AI_MnistDatasetFree(&mnist);
}

void ai_onednn_lenet5()
{
    size_t num_epochs = 1000;
    size_t N = 16;
    float learning_rate = 0.01f;

    ai_dnnl_convolutional_layer_create_info_t c1_info;
    c1_info.OC = 6;
    c1_info.KH = 5;
    c1_info.KW = 5;
    c1_info.SH = 1;
    c1_info.SW = 1;
    c1_info.PT = 0;
    c1_info.PL = 0;
    c1_info.PB = 0;
    c1_info.PR = 0;
    c1_info.learning_rate = learning_rate;
    c1_info.weight_init = ai_dnnl_convolutional_layer_weight_init_kind_xavier;
    c1_info.bias_init = ai_dnnl_convolutional_layer_bias_init_kind_zeros;

    ai_dnnl_activation_layer_create_info_t a1_info;
    a1_info.activation = ai_dnnl_activation_kind_tanh;
    
    ai_dnnl_pooling_layer_create_info_t p1_info;
    p1_info.pooling_kind = ai_dnnl_pooling_avg;
    p1_info.KH = 2;
    p1_info.KW = 2;
    p1_info.SH = 2;
    p1_info.SW = 2;
    p1_info.PT = 0;
    p1_info.PL = 0;
    p1_info.PB = 0;
    p1_info.PR = 0;

    ai_dnnl_convolutional_layer_create_info_t c2_info;
    c2_info.OC = 16;
    c2_info.KH = 5;
    c2_info.KW = 5;
    c2_info.SH = 1;
    c2_info.SW = 1;
    c2_info.PT = 0;
    c2_info.PL = 0;
    c2_info.PB = 0;
    c2_info.PR = 0;
    c2_info.learning_rate = learning_rate;
    c2_info.weight_init = ai_dnnl_convolutional_layer_weight_init_kind_xavier;
    c2_info.bias_init = ai_dnnl_convolutional_layer_bias_init_kind_zeros;

    ai_dnnl_activation_layer_create_info_t a2_info;
    a2_info.activation = ai_dnnl_activation_kind_tanh;

    ai_dnnl_pooling_layer_create_info_t p2_info;
    p2_info.pooling_kind = ai_dnnl_pooling_avg;
    p2_info.KH = 2;
    p2_info.KW = 2;
    p2_info.SH = 2;
    p2_info.SW = 2;
    p2_info.PT = 0;
    p2_info.PL = 0;
    p2_info.PB = 0;
    p2_info.PR = 0;

    ai_dnnl_linear_layer_create_info_t l3_info;
    l3_info.OC = 120;
    l3_info.learning_rate = learning_rate;
    l3_info.weight_init = ai_dnnl_linear_layer_weight_init_kind_xavier;
    l3_info.bias_init = ai_dnnl_linear_layer_bias_init_kind_zeros;
    l3_info.allow_reorder = 1;

    ai_dnnl_activation_layer_create_info_t a3_info;
    a3_info.activation = ai_dnnl_activation_kind_tanh;

    ai_dnnl_linear_layer_create_info_t l4_info;
    l4_info.OC = 84;
    l4_info.learning_rate = learning_rate;
    l4_info.weight_init = ai_dnnl_linear_layer_weight_init_kind_xavier;
    l4_info.bias_init = ai_dnnl_linear_layer_bias_init_kind_zeros;
    l4_info.allow_reorder = 1;


    ai_dnnl_activation_layer_create_info_t a4_info;
    a4_info.activation = ai_dnnl_activation_kind_tanh;

    ai_dnnl_linear_layer_create_info_t l5_info;
    l5_info.OC = 10;
    l5_info.learning_rate = learning_rate;
    l5_info.weight_init = ai_dnnl_linear_layer_weight_init_kind_xavier;
    l5_info.bias_init = ai_dnnl_linear_layer_bias_init_kind_zeros;
    l5_info.allow_reorder = 1;

    ai_dnnl_activation_layer_create_info_t a5_info;
    a5_info.activation = ai_dnnl_activation_kind_tanh;

    ai_dnnl_layer_create_info_t create_infos[] = {
        { ai_dnnl_layer_kind_convolutional, &c1_info },
        { ai_dnnl_layer_kind_activation, &a1_info },
        { ai_dnnl_layer_kind_pooling, &p1_info },
        { ai_dnnl_layer_kind_convolutional, &c2_info },
        { ai_dnnl_layer_kind_activation, &a2_info },
        { ai_dnnl_layer_kind_pooling, &p2_info },
        { ai_dnnl_layer_kind_linear, &l3_info},
        { ai_dnnl_layer_kind_activation, &a3_info},
        { ai_dnnl_layer_kind_linear, &l4_info},
        { ai_dnnl_layer_kind_activation, &a4_info},
        { ai_dnnl_layer_kind_linear, &l5_info},
        { ai_dnnl_layer_kind_activation, &a5_info},
    };

    ai_input_dims_t input_dims = { N, 1, 32, 32 };

    ai_dnnl_model_t* model;
    AI_MnistDataset mnist;

    uint32_t status = ai_dnnl_model_create(&model, &input_dims, 12, create_infos, ai_dnnl_loss_mse);
    AI_MnistDatasetLoad(&mnist, "./datasets/fashion_mnist", 2);

    printf("model created: %d\n", status);

    status = ai_dnnl_model_train(model, 60000, mnist.train_images, mnist.train_labels, 10000, mnist.test_images, mnist.test_labels, num_epochs, dnnl_train_callback);
    printf("status: %d\n", status);

}

int main()
{
    ai_onednn_lenet5();
}
