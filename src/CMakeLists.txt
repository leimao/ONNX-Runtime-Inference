cmake_minimum_required(VERSION 3.13)

project(ONNX_Runtime_Inference VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

add_executable(inference_cuda inference_cuda.cpp)
target_include_directories(inference_cuda PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(inference_cuda PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})

add_executable(inference_cpu inference_cpu.cpp)
target_include_directories(inference_cpu PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(inference_cpu PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})