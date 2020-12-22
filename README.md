# ONNX Runtime Inference

## Introduction

ONNX Runtime inference C++ example for image classification using CPU and CUDA.

## Usages

### Build Docker Image

```bash
$ docker build -f docker/onnxruntime-cuda.Dockerfile --no-cache --tag=onnxruntime-cuda:1.6.0 .
```

### Run Docker Container

```bash
$ docker run -it --rm --gpus device=0 -v $(pwd):/mnt onnxruntime-cuda:1.6.0
```

### Build Example

```bash
$ cmake -B build
$ cmake --build build --config Release --parallel 8
```

### Run Example

```bash
$ cd build/src/
$ ./inference_cpu 
Number of input nodes: 1
Number of output nodes: 1
float
[1, 3, 224, 224]
float
[1, 1000]
Predicted Label ID: 92
Predicted Label: n01828970 bee eater
Uncalibrated Confidence: 0.996137
Minimum Latency: 7.51[ms]
```

```bash
$ cd build/src/
$ ./inference_cuda 
Number of input nodes: 1
Number of output nodes: 1
float
[1, 3, 224, 224]
float
[1, 1000]
Predicted Label ID: 92
Predicted Label: n01828970 bee eater
Uncalibrated Confidence: 0.996137
Minimum Latency: 1.03[ms]
```

### References
