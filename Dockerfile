# ==========================================
# STAGE 1: Build Environment
# ==========================================
# Build the base image with: jetson-containers build --skip-tests all cuda
# 
FROM cuda:r39.2.tegra-aarch64-cu132-24.04-cuda AS downloader

WORKDIR /downloads

RUN wget -O tensorrt.deb https://developer.download.nvidia.com/compute/tensorrt/10.16.1/local_installers/nv-tensorrt-local-repo-ubuntu2404-10.16.1-cuda-13.2_1.0-1_arm64.deb

RUN wget -O deepstream.deb https://github.com/NVIDIA/DeepStream/releases/download/v9.2.0/deepstream-9.2_9.2.0-1_arm64.deb

FROM cuda:r39.2.tegra-aarch64-cu132-24.04-cuda 

WORKDIR /downloads

RUN --mount=type=bind,from=downloader,source=/downloads,target=/downloads \
    dpkg -i tensorrt.deb \
    && cp /var/*tensorrt*/*-keyring.gpg /usr/share/keyrings/ \
    && apt update \
    && apt install -y python3-libnvinfer-lean

RUN apt update && apt install -y --no-install-recommends \
    libssl3 \
    libssl-dev \
    libcurl4-openssl-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer-plugins-base1.0-dev \
    libgstrtspserver-1.0-0 \
    libjansson4 \
    libyaml-cpp-dev \
    libmosquitto1

# RUN --mount=type=bind,from=downloader,source=/downloads,target=/downloads \
#     apt install -y ./deepstream.deb
# 
RUN wget https://github.com/NVIDIA/DeepStream/releases/download/v9.2.0/deepstream-binaries-aarch64_9.2.0_arm64.deb \
    && apt install -y ./deepstream*.deb

# docker build -f Dockerfile -t mv_app:latest .