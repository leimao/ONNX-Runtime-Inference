FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG ONNX_VERSION=1.17.0
ARG ONNXRUNTIME_VERSION=1.21.0
ARG OPENCV_VERSION=4.11.0
# ARG CMAKE_VERSION=4.0.0
ARG CMAKE_VERSION=3.31.0
ARG NUM_JOBS=4

ARG PYTHON_VENV_PATH="/python/venv"

ENV DEBIAN_FRONTEND=noninteractive

# Install package dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        locales \
        locales-all \
        python3-full \
        wget \
        git && \
    apt-get clean

# System locale
# Important for UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

RUN mkdir -p ${PYTHON_VENV_PATH} && \
    python3 -m venv ${PYTHON_VENV_PATH}

ENV PATH=${PYTHON_VENV_PATH}/bin:$PATH

RUN cd ${PYTHON_VENV_PATH}/bin && \
    pip install --upgrade pip setuptools wheel

# Install CMake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    bash cmake-${CMAKE_VERSION}-linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm -rf /tmp/*

# Install OpenCV
# OpenCV-Python dependencies
RUN apt-get update && apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
RUN apt-get update && apt-get install -y libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
RUN apt-get update && apt-get install -y libgtk-3-dev 
RUN apt-get update && apt-get install -y libpng-dev libopenexr-dev libtiff-dev libwebp-dev
RUN apt-get update && apt-get install -y unzip

RUN cd /tmp && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    wget -O opencv_extra.zip https://github.com/opencv/opencv_extra/archive/${OPENCV_VERSION}.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip && \
    unzip opencv_extra.zip
RUN cd /tmp && \
    mkdir -p build && cd build && \
    cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-${OPENCV_VERSION}/modules \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DBUILD_PNG=OFF \
        -DBUILD_TIFF=OFF \
        -DBUILD_TBB=OFF \
        -DBUILD_JPEG=OFF \
        -DBUILD_JASPER=OFF \
        -DBUILD_ZLIB=OFF \
        -DBUILD_EXAMPLES=ON \
        -DBUILD_JAVA=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=ON \
        -DWITH_OPENCL=OFF \
        -DWITH_OPENMP=OFF \
        -DWITH_FFMPEG=ON \
        -DWITH_GSTREAMER=OFF \
        -DWITH_GSTREAMER_0_10=OFF \
        -DWITH_CUDA=ON \
        -DWITH_GTK=ON \
        -DWITH_VTK=OFF \
        -DWITH_TBB=ON \
        -DWITH_1394=OFF \
        -DWITH_OPENEXR=OFF \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
        -DCUDA_ARCH_BIN='7.0 7.5 8.0' \
        -DCUDA_ARCH_PTX="" \
        -DINSTALL_C_EXAMPLES=ON \
        -DINSTALL_TESTS=OFF \
        -DOPENCV_TEST_DATA_PATH=../opencv_extra-${OPENCV_VERSION}/testdata \
        ../opencv-${OPENCV_VERSION} && \
    cmake --build . --parallel ${NUM_JOBS} && \
    make install && \
    rm -rf /tmp/*

# Install ONNX Runtime
RUN pip install numpy==2.2.4 psutil==7.0.0 pytest==8.3.5 onnx==${ONNX_VERSION}
RUN cd /tmp && \
    git clone --recursive --branch v${ONNXRUNTIME_VERSION} https://github.com/Microsoft/onnxruntime
RUN cd /tmp && \
    cd onnxruntime && \
    ./build.sh \
        --allow_running_as_root \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr/lib/x86_64-linux-gnu/ \
        --use_cuda \
        # --use_tensorrt \
        # --tensorrt_home /usr/lib/x86_64-linux-gnu/ \
        --config RelWithDebInfo \
        --build_shared_lib \
        # Somehow --build_wheel cannot be used in Docker build because NumPy cannot be found.
        # However, if this command is run in a Docker container, it works.
        # --build_wheel \
        --skip_tests \
        --parallel ${NUM_JOBS} && \
    cd build/Linux/RelWithDebInfo && \
    make install && \
    # pip install dist/* && \
    rm -rf /tmp/*

