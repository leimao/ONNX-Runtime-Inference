cmake_minimum_required(VERSION 3.13)

project(ONNX_Runtime_Inference VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

add_executable(inference inference.cpp)
target_include_directories(inference PRIVATE ${ONNX_RUNTIME_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(inference PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})