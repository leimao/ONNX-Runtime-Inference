# ONNX Runtime Inference

## Introduction

ONNX Runtime C++ and Python inference example for image classification using CPU and CUDA.

## Usages

### C++ Inference

#### Build Docker Image

```bash
$ docker build -f docker/onnxruntime-cuda.Dockerfile --no-cache --tag onnxruntime-cuda:1.21.0 .
```

#### Run Docker Container

```bash
$ docker run -it --rm --gpus device=0 -v $(pwd):/mnt -w /mnt onnxruntime-cuda:1.21.0
```

#### Build Example

```bash
$ cmake -B build
$ cmake --build build --config Release --parallel
```

#### Run Example

```bash
$ cd build/src/
$ ./inference  --use_cpu
Inference Execution Provider: CPU
Number of Input Nodes: 1
Number of Output Nodes: 1
Input Name: data
Input Type: float
Input Dimensions: [1, 3, 224, 224]
Output Name: squeezenet0_flatten0_reshape0
Output Type: float
Output Dimensions: [1, 1000]
Predicted Label ID: 92
Predicted Label: n01828970 bee eater
Uncalibrated Confidence: 0.996137
Minimum Inference Latency: 7.45 ms
```

```bash
$ cd build/src/
$ ./inference  --use_cuda
Inference Execution Provider: CUDA
Number of Input Nodes: 1
Number of Output Nodes: 1
Input Name: data
Input Type: float
Input Dimensions: [1, 3, 224, 224]
Output Name: squeezenet0_flatten0_reshape0
Output Type: float
Output Dimensions: [1, 1000]
Predicted Label ID: 92
Predicted Label: n01828970 bee eater
Uncalibrated Confidence: 0.996137
Minimum Inference Latency: 0.98 ms
```

### Python Inference

#### Build Docker Image

```bash
$ docker build -f docker/onnxruntime-cuda-python.Dockerfile --no-cache --tag onnxruntime-cuda-python:1.21.0 .
```

#### Run Docker Container

```bash
$ docker run -it --rm --gpus device=0 -v $(pwd):/mnt -w /mnt onnxruntime-cuda-python:1.21.0
```

#### Run Example

```bash
$ python python/inference.py
Predicted Label ID: 92
Predicted Label: n01828970 bee eater
```

## References

- [ONNX Runtime C++ Inference](https://leimao.github.io/blog/ONNX-Runtime-CPP-Inference/)
- [ONNX Runtime Python Inference](https://leimao.github.io/blog/ONNX-Runtime-Python-Inference/)
