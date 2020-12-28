# ONNX Runtime Inference

## Introduction

ONNX Runtime C++ inference example for image classification using CPU and CUDA.

## Dependencies

* CMake 3.16.8
* ONNX Runtime 1.6.0
* OpenCV 4.5.0

## Usages

### Build Docker Image

```bash
$ docker build -f docker/onnxruntime-cuda.Dockerfile --no-cache --tag=onnxruntime-cuda:0.0.1 .
```

### Run Docker Container

```bash
$ docker run -it --rm --gpus device=0 -v $(pwd):/mnt onnxruntime-cuda:0.0.1
```

Debug mode

```bash
$ xhost +
$ docker run -it --rm --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --gpus device=0 --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/mnt onnxruntime-cuda:0.0.1
$ xhost -
```

### Build Example

```bash
$ cmake -B build
$ cmake --build build --config Release --parallel
```

### Run Example

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

## References

* [ONNX Runtime C++ Inference](https://leimao.github.io/blog/ONNX-Runtime-CPP-Inference/)
