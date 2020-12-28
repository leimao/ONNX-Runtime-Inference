FROM nvcr.io/nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
# FROM nvcr.io/nvidia/tensorrt:20.09-py3

ARG OPENCV_VERSION=4.5.1
ARG ONNXRUNTIME_VERSION=1.6.0
ARG CMAKE_VERSION=3.19.2
ARG NUM_JOBS=12

ENV DEBIAN_FRONTEND noninteractive

# Install package dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        wget \
        git \
        curl \
        libjpeg-dev \
        libpng-dev \
        language-pack-en \
        locales \
        locales-all \
        python3 \
        python3-py \
        python3-dev \
        python3-pip \
        python3-numpy \
        python3-pytest \
        python3-setuptools \
        libprotobuf-dev \
        protobuf-compiler \
        zlib1g-dev \
        swig \
        vim \
        gdb \
        valgrind \
        libsm6 \
        libxext6 \
        libxrender-dev \
        cmake \
        unzip \
        libcanberra-gtk-module \
        libcanberra-gtk3-module \
        sudo

RUN cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    ln -s /usr/bin/pip3 pip && \
    pip install --upgrade pip setuptools wheel

# System locale
# Important for UTF-8
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

# Install CMake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh && \
    chmod +x cmake-${CMAKE_VERSION}-Linux-x86_64.sh && \
    ./cmake-${CMAKE_VERSION}-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license
# RUN rm -rf /tmp/*

# Install OpenCV
# OpenCV-Python dependencies
RUN apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
RUN apt-get install -y libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
RUN apt-get install -y libgtk-3-dev 
RUN apt-get install -y libpng-dev libopenexr-dev libtiff-dev libwebp-dev libdc1394-22-dev
RUN apt-get install -y libv4l-dev
RUN apt-get install -y install ffmpeg

RUN cd /tmp && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    wget -O opencv_extra.zip https://github.com/opencv/opencv_extra/archive/${OPENCV_VERSION}.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip && \
    unzip opencv_extra.zip && \
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
        -DWITH_GSTREAMER=ON \
        -DVIDEOIO_PLUGIN_LIST=gstreamer \
        -DWITH_GSTREAMER_0_10=ON \
        -DWITH_CUDA=ON \
        -DWITH_GTK=ON \
        -DWITH_VTK=OFF \
        -DWITH_TBB=ON \
        -DWITH_V4L=ON \
        -DWITH_1394=ON \
        -DWITH_OPENEXR=OFF \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
        -DCUDA_ARCH_BIN='3.0 3.5 5.0 6.0 6.2 7.0 7.5' \
        -DCUDA_ARCH_PTX="" \
        -DINSTALL_C_EXAMPLES=ON \
        -DINSTALL_TESTS=OFF \
        -DOPENCV_TEST_DATA_PATH=../opencv_extra-${OPENCV_VERSION}/testdata \
        ../opencv-${OPENCV_VERSION} && \
    cmake --build . --parallel ${NUM_JOBS} && \
    make install
# RUN rm -rf /tmp/*

# Install ONNX Runtime
RUN pip install pytest==6.2.1 onnx==1.8.0
RUN cd /tmp && \
    git clone --recursive --branch v${ONNXRUNTIME_VERSION} https://github.com/Microsoft/onnxruntime && \
    cd onnxruntime && \
    ./build.sh \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr/lib/x86_64-linux-gnu/ \
        --use_cuda \
        # --use_tensorrt \
        # --tensorrt_home /usr/lib/x86_64-linux-gnu/ \
        --config RelWithDebInfo \
        --build_shared_lib \
        --build_wheel \
        --skip_tests \
        --parallel ${NUM_JOBS} && \
    cd build/Linux/RelWithDebInfo && \
    make install && \
    pip install dist/*
# RUN rm -rf /tmp/*

