# Neuralnet
Library for training and of neural networks implemented in C. Includes two backends:
* naive implementation where all primitives are implemented from scratch
* [Intel oneDNN](https://github.com/oneapi-src/oneDNN) backend (only cpu)

Features:
* fully-connected, convolution, pooling and batchnorm layers
* optimizers RMSProp, Adam and SGD
* different activation functions, dropout
* weight regularization: l1 and l2
* data augmentation techniques: image flips, random crop, color augmentations

## Examples
Provided example models with validation accuracy. Can be found in the [examples](examples) folder
| example | mnist | fashion_mnist |
| ------- | ------| ------------- |
| logistic regression | 90.6% | 79.36% |
| two layer MLP (300 hidden units) | 98.43% | 86.81% |
| LeNet-5 | 99.13% | 90.74% |
| Conv-Bn-Relu CNN (7M parameters) | - | 94.60% |

## How to run
Compile and run example applications in one step using the naive backend. Run the [mlp](examples/mlp.c) example:
```bash
make TARGET=examples/mlp BACKEND=naive run
```
Naive backend can be configured to use AVX2 to vectorize matrix products and vector operations:
```bash
make TARGET=examples/mlp BACKEND=naive USE_AVX=1 run
```

## How to use oneDNN backend
Requires [Intel oneDNN](https://github.com/oneapi-src/oneDNN) to be installed as a shared library. A [script](scripts/install-onednn.sh) is provided which compiles the specific revision of oneDNN used for development from source.

When compiling and running, the path to the oneDNN installation must be configured in ```ONEDNN_ROOT_DIR```:
```bash
make TARGET=examples/mlp \
     BACKEND=onednn \
     ONEDNN_ROOT_DIR=[path to onednn installation] \
     USE_AVX=1 run
```
or to run without make (if example is already compiled)
```bash
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:[path to shared library (libdnnl.so)] \
    ./examples/mlp
```