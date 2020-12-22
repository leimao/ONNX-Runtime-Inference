cmake_minimum_required(VERSION 3.13)

project(ONNX_Runtime_Examples VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(OpenCV REQUIRED)
find_path(ONNX_RUNTIME_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)
message(${ONNX_RUNTIME_INCLUDE_DIR})
#include_directories(/usr/local/include)
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)
message(${ONNX_RUNTIME_LIB})

add_subdirectory(src)
