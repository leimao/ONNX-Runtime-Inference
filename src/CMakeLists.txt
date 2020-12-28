cmake_minimum_required(VERSION 3.13)

project(ONNX_Runtime_Inference VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(Threads REQUIRED)


add_executable(inference inference.cpp commons.h)
target_include_directories(inference PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(inference PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})

add_executable(inference_camera inference_camera.cpp commons.h)
target_include_directories(inference_camera PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(inference_camera PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES} Threads::Threads)

add_executable(camera_fps camera_fps.cpp)
target_include_directories(camera_fps PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(camera_fps PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES} Threads::Threads swscale avformat avcodec avutil)
