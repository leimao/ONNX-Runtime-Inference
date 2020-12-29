#include <onnxruntime_cxx_api.h>

#include <fstream>
#include <numeric>
#include <vector>
#include <stdexcept>
#include <queue>
#include <mutex>
#include <condition_variable>

/**
 * @brief Compute the product of a numeric vector.
 * @tparam T 
 * @param v 
 * @return T 
 */
template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

/**
 * @brief Print ONNX tensor data type
 * https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L93
 * @param os
 * @param type
 * @return std::ostream&
 */
std::ostream& operator<<(std::ostream& os,
                         const ONNXTensorElementDataType& type)
{
    switch (type)
    {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            os << "undefined";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            os << "float";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            os << "uint8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            os << "int8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            os << "uint16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            os << "int16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            os << "int32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            os << "int64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            os << "std::string";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            os << "bool";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            os << "float16";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            os << "double";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            os << "uint32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            os << "uint64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            os << "float real + float imaginary";
            break;
        case ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            os << "double real + float imaginary";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            os << "bfloat16";
            break;
        default:
            break;
    }

    return os;
}

/**
 * @brief Read labels from a label file.
 * @param labelFilepath 
 * @return std::vector<std::string> 
 */
std::vector<std::string> readLabels(const std::string& labelFilepath)
{
    std::vector<std::string> labels;
    std::string line;
    std::ifstream fp(labelFilepath);
    if (!fp.is_open())
    {
        throw std::runtime_error{"Label file was not found."};
    }
    while (std::getline(fp, line))
    {
        labels.push_back(line);
    }
    return labels;
}

/**
 * @brief Preprocess a HWC BGR uint8 image to a resized CHW RGB float32 image.
 * @param imageBGR 
 * @param height 
 * @param width 
 * @return cv::Mat 
 */
cv::Mat preprocessImage(cv::Mat& imageBGR, const int height, const int width)
{
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;

    cv::resize(imageBGR, resizedImageBGR,
               cv::Size(height, width),
               cv::InterpolationFlags::INTER_CUBIC);
    cv::cvtColor(resizedImageBGR, resizedImageRGB,
                 cv::ColorConversionCodes::COLOR_BGR2RGB);
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

    cv::Mat channels[3];
    cv::split(resizedImage, channels);
    // Normalization per channel
    // Normalization parameters obtained from
    // https://github.com/onnx/models/tree/master/vision/classification/squeezenet
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;
    cv::merge(channels, 3, resizedImage);
    // HWC to CHW
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);

    return preprocessedImage;
}

template <typename T>
Ort::Value createIOTensors(std::vector<T>& tensorValues, const std::vector<int64_t>& dims)
{
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<T>(memoryInfo, tensorValues.data(), tensorValues.size(), dims.data(), dims.size());
    
    return tensor;
}

template <typename T>
void getPrediction(const std::vector<T>& outputTensorValues, int& predId, float& confidence)
{
    float activation = 0;
    float maxActivation = std::numeric_limits<float>::lowest();
    float expSum = 0;
    for (int i = 0; i < outputTensorValues.size(); i++)
    {
        activation = outputTensorValues.at(i);
        expSum += std::exp(activation);
        if (activation > maxActivation)
        {
            predId = i;
            maxActivation = activation;
        }
    }
}

float measureInferenceLatency(Ort::Session& session, const std::vector<const char*>& inputNames, const std::vector<Ort::Value>& inputTensors, const std::vector<const char*>& outputNames, std::vector<Ort::Value>& outputTensors, const int numTests)
{
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    for (int i = 0; i < numTests; i++)
    {
        session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                    inputTensors.data(), 1, outputNames.data(),
                    outputTensors.data(), 1);
    }
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();

    float latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / static_cast<float>(numTests);

    return latency;
}

Ort::Session createInferenceSession(Ort::Env& env, const std::string& modelFilepath, bool useCUDA)
{
    Ort::SessionOptions sessionOptions;
    // Session configurations
    // https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/docs/ONNX_Runtime_Perf_Tuning.md#default-cpu-execution-provider-mlas
    // Use default settings
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetInterOpNumThreads(1);
    // ExecutionMode::ORT_PARALLEL or ExecutionMode::ORT_SEQUENTIAL
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    if (useCUDA)
    {
        // Using CUDA backend
        // https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/providers/cuda/cuda_provider_factory.h#L13
        OrtStatus* status =
            OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
    }

    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
    // removals) ORT_ENABLE_EXTENDED -> To enable extended optimizations
    // (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible optimizations
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);

    return session;
}

/**
 * @brief A thread-safe fixed-size queue with a throughput counter.
 * This might be useful for some memory-constrained platforms.
 * https://stackoverflow.com/questions/15278343/c11-thread-safe-queue
 * @tparam T 
 */
// It is unnecessary to inherit.
// template <typename T>
// class FixedQueue : public std::queue<T>
// {
// public:

//     FixedQueue() : mCounter{0}, mMaxSize{0} {}
//     FixedQueue(size_t maxSize) : mMaxSize{maxSize} {}

//     void push(const T& value)
//     {
//         std::lock_guard<std::mutex> lock(this->mMutex);
//         // Remove the first element if the queue reaches maximum size.
//         if ((this->mMaxSize > 0) && (std::queue<T>::size() == this->mMaxSize))
//         {
//             std::queue<T>::pop();
//         }
//         std::queue<T>::push(value);
//         this->mCounter += 1;
//         if (this->mCounter == 1)
//         {
//             this->mTickMeter.reset();
//             // We start counting from the second frame.
//             this->mTickMeter.start();
//         }
//         this->mCV.notify_one();
//     }

//     void enqueue(const T& value)
//     {
//         this->push();
//     }

//     bool empty() const
//     {
//         std::lock_guard<std::mutex> lock(this->mMutex);
//         bool isEmpty = std::queue<T>::empty();
//         return isEmpty;
//     }

//     void pop()
//     {
//         std::lock_guard<std::mutex> lock(this->mMutex);
//         std::queue<T>::pop();
//     }

//     T& front()
//     {
//         std::lock_guard<std::mutex> lock(this->mMutex);
//         T& value = std::queue<T>::front();
//         return value;
//     }

//     T dequeue()
//     {
//         // We use std::unique_lock instead of std::lock_guard here 
//         // because we will release the lock temporarily later if the queue is empty.
//         std::unique_lock<std::mutex> lock(this->mMutex);
//         // https://en.cppreference.com/w/cpp/thread/condition_variable/wait
//         this->mCV.wait(lock, std::queue<T>::empty);
        
//         T value = std::queue<T>::front();
//         std::queue<T>::pop();

//         return value;
//     }

//     float getFPS()
//     {
//         this->mTickMeter.stop();
//         float fps = static_cast<float>(this->mCounter) / this->mTickMeter.getTimeSec();
//         this->mTickMeter.start();
//         return fps;
//     }

//     void clear()
//     {
//         std::lock_guard<std::mutex> lock(this->mMutex);
//         while (!std::queue<T>::empty())
//         {
//             std::queue<T>::pop();
//         }
//     }
    
// private:

//     cv::TickMeter mTickMeter;
//     std::mutex mMutex;
//     // If mMaxSize <= 0
//     // FixedQueue object does not have size limit.
//     size_t mMaxSize;
//     unsigned int mCounter;
//     std::condition_variable mCV;

// };

/**
 * @brief A thread-safe fixed-size queue with a throughput counter.
 * This should be equivalent to a ring-queue or a ring-buffer.
 * This might be useful for some memory-constrained platforms.
 * https://stackoverflow.com/questions/15278343/c11-thread-safe-queue
 * @tparam T 
 */
template <typename T>
class FixedQueue
{
public:

    FixedQueue() : mCounter{0}, mWarmUp{1}, mMaxSize{0} {}
    FixedQueue(size_t maxSize) : mCounter{0}, mWarmUp{1}, mMaxSize{maxSize} {}

    /**
     * @brief Insert a value to the FixedQueue object in a thread safe way.
     * @param value 
     */
    void enqueue(const T& value)
    {
        std::lock_guard<std::mutex> lock(this->mMutex);
        // Remove the first element if the queue reaches maximum size.
        if ((this->mMaxSize > 0) && (this->mQueue.size() == this->mMaxSize))
        {
            this->mQueue.pop();
        }
        this->mQueue.push(value);
        this->mCounter += 1;
        if (this->mCounter == this->mWarmUp)
        {
            this->mTickMeter.reset();
            // We start counting from the second frame.
            this->mTickMeter.start();
        }
        this->mCV.notify_one();
    }

    /**
     * @brief Retrieve a value from the FixedQueue object in a thread safe way.
     * @param value 
     */
    T dequeue()
    {
        // We use std::unique_lock instead of std::lock_guard here 
        // because we will release the lock temporarily later if the queue is empty.
        std::unique_lock<std::mutex> lock(this->mMutex);
        // https://en.cppreference.com/w/cpp/thread/condition_variable/wait
        while(this->mQueue.empty())
        {
            this->mCV.wait(lock);
        }

        T value = this->mQueue.front();
        this->mQueue.pop();

        return value;
    }

    /**
     * @brief Get the throughput of the FixedQueue object.
     * @return float 
     */
    float getThroughput()
    {
        this->mTickMeter.stop();
        long numValidCounts = this->mCounter - this->mWarmUp;
        float throughput = 0;
        if (numValidCounts > 0)
        {
            throughput = numValidCounts / this->mTickMeter.getTimeSec();
        }
        std::cout << this->mCounter << ", " << this->mTickMeter.getTimeSec() << std::endl;
        this->mTickMeter.start();
        return throughput;
        // this->mTickMeter.stop();
        // double fps = (this->mCounter - this->mWarmUp) / this->mTickMeter.getTimeSec();
        
        // std::cout << this->mCounter << ", " << this->mTickMeter.getTimeSec() << std::endl;
        // this->mTickMeter.start();
        // return static_cast<float>(fps);
    }

    /**
     * @brief Clear the FixedQueue object.
     */
    void clear()
    {
        std::lock_guard<std::mutex> lock(this->mMutex);
        this->mQueue = std::queue<T>();
        this->mCounter = 0;
    }
    
private:

    std::queue<T> mQueue;
    cv::TickMeter mTickMeter;
    std::mutex mMutex;
    // If mMaxSize <= 0
    // FixedQueue object does not have size limit.
    size_t mMaxSize;
    long mCounter;
    int mWarmUp;
    std::condition_variable mCV;
};

