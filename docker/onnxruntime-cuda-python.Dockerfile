FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG ONNX_VERSION=1.17.0
ARG ONNXRUNTIME_VERSION=1.21.0

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

RUN pip install Pillow==11.2.0 onnx==${ONNX_VERSION} onnxruntime-gpu==${ONNXRUNTIME_VERSION}
