# ONNX Runtime Inference


```
wget https://github.com/onnx/models/raw/master/vision/classification/squeezenet/model/squeezenet1.1-7.onnx
```

```
cmake -B build
cmake --build build --config Release
cmake --build build --target install --config Release --parallel 8
```

docker build -f docker/onnxruntime-cuda.Dockerfile --no-cache --tag=onnxruntime-cuda:10.2 .