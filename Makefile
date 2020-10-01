
ONEDNN_DIR = D:/dev/tools/oneDNN_msbuild

CFLAGS = -Ofast -march=haswell -mavx -mavx2
INCLUDE = -I$(ONEDNN_DIR)/include
LDFLAGS =  $(ONEDNN_DIR)/bin/dnnl.dll
SRC = main.c  ai_util/ai_gradient_clipping.c ai_util/ai_loss.c ai_util/ai_math.c ai_util/ai_random.c ai_util/ai_weight_init.c ai_datasets/ai_mnist.c ai_layer/ai_base_layer.c \
	ai_layer/ai_input_layer.c ai_layer/ai_linear_layer.c ai_layer/ai_activation_layer.c ai_layer/ai_convolutional_layer.c ai_layer/ai_pooling_layer.c ai_layer/ai_dropout_layer.c \
	ai_net.c \
	ai_layer/ai_dnnl_base_layer.c ai_layer/ai_dnnl_linear_layer.c ai_layer/ai_dnnl_activation_layer.c ai_layer/ai_dnnl_reorder_layer.c ai_util/ai_dnnl_util.c \
	ai_layer/ai_dnnl_input_layer.c ai_util/ai_dnnl_loss.c ai_dnnl_model.c ai_util/ai_dnnl_reorder.c ai_layer/ai_dnnl_convolutional_layer.c ai_layer/ai_dnnl_pooling_layer.c \
	ai_model_desc.c

main.exe: $(SRC)
	gcc $(CFLAGS) $(INCLUDE) $(SRC) $(LDFLAGS) -o main.exe

clean:
	del main.exe

rebuild:
	make clean && make