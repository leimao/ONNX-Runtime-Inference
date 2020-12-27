
// https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
// https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_cxx_api.h
#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "commons.h"

int main(int argc, char* argv[])
{
    bool useCUDA{true};
    const char* useCUDAFlag = "--use_cuda";
    const char* useCPUFlag = "--use_cpu";
    if (argc == 1)
    {
        useCUDA = false;
    }
    else if ((argc == 2) && (strcmp(argv[1], useCUDAFlag) == 0))
    {
        useCUDA = true;
    }
    else if ((argc == 2) && (strcmp(argv[1], useCPUFlag) == 0))
    {
        useCUDA = false;
    }
    else if (argc == 2)
    {
        throw std::runtime_error{"Unsupported argument."};
    }
    else
    {
        throw std::runtime_error{"Too many arguments."};
    }

    if (useCUDA)
    {
        std::cout << "Inference Execution Provider: CUDA" << std::endl;
    }
    else
    {
        std::cout << "Inference Execution Provider: CPU" << std::endl;
    }

    std::string instanceName{"image-classification-inference"};
    std::string modelFilepath{"../../data/models/squeezenet1.1-7.onnx"};
    std::string imageFilepath{
        "../../data/images/european-bee-eater-2115564_1920.jpg"};
    std::string labelFilepath{"../../data/labels/synset.txt"};

    std::vector<std::string> labels{readLabels(labelFilepath)};

    // https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L123
    Ort::Env env{OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str()};

    Ort::Session session = createInferenceSession(env, modelFilepath, useCUDA);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;

    const char* inputName = session.GetInputName(0, allocator);
    std::cout << "Input Name: " << inputName << std::endl;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::cout << "Input Type: " << inputType << std::endl;

    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    std::cout << "Input Dimensions: " << inputDims << std::endl;

    const char* outputName = session.GetOutputName(0, allocator);
    std::cout << "Output Name: " << outputName << std::endl;

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    std::cout << "Output Type: " << outputType << std::endl;

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    std::cout << "Output Dimensions: " << outputDims << std::endl;

    cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
    cv::Mat preprocessedImage = preprocessImage(imageBGR, inputDims.at(2), inputDims.at(3));

    size_t inputTensorSize = vectorProduct(inputDims);
    std::vector<float> inputTensorValues(inputTensorSize);
    inputTensorValues.assign(preprocessedImage.begin<float>(),
                             preprocessedImage.end<float>());

    size_t outputTensorSize = vectorProduct(outputDims);
    assert(("Output tensor size should equal to the label set size.",
            labels.size() == outputTensorSize));
    std::vector<float> outputTensorValues(outputTensorSize);

    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames{outputName};
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    inputTensors.push_back(createIOTensors(inputTensorValues, inputDims));
    outputTensors.push_back(createIOTensors(outputTensorValues, outputDims));

    // https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_cxx_api.h#L353
    session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), 1, outputNames.data(),
                outputTensors.data(), 1);

    int predId = 0;
    float confidence = 0;
    getPrediction(outputTensorValues, predId, confidence);

    std::cout << "Predicted Label ID: " << predId << std::endl;
    std::cout << "Predicted Label: " << labels.at(predId) << std::endl;
    std::cout << "Uncalibrated Confidence: " << confidence << std::endl;

    // Measure latency
    int numTests{100};
    float latency = measureInferenceLatency(session, inputNames, inputTensors, outputNames, outputTensors, numTests);
    std::cout << "Minimum Inference Latency: " << latency << " ms" << std::endl;
}