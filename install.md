# Installation


## Clean resource

clean a bit of ram
```bash
sudo sysctl -w vm.drop_caches=3
```

## Python venv

enable `include-system-site-packages = true` so the venv can see TensorRT and Gstreamer binaries

create venv with uv
```bash
uv venv --system-site-packages
```

## TensorRT

TensorRT comes preinstalled when flashing Jetpack. 

But if required it can be installed from a .deb, a python package, and other sources.

Source: https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/installing-tensorrt/overview.html

on the jetson host, easy install is. but for container you need to follow the instructions
```bash
uv add tensorrt-cu13

```

## Deepstream

source: https://docs.nvidia.com/deepstream/text/DS_Installation

```bash
# install dependencies
sudo apt install \
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

# download debian package - can install also from a tarball 
wget https://github.com/NVIDIA/DeepStream/releases/download/v9.2.0/deepstream-9.2_9.2.0-1_arm64.deb
sudo apt install ./deepstream-9.2_9.2.0-1_arm64.deb

# delete the deb
rm -rf deepstream-9.2_9.2.0-1_arm64.deb

```


avoid error messages on logs with:
```bash
sudo systemctl stop nvargus-daemon.service
sudo systemctl disable nvargus-daemon.service
```


## YOLO26 models

https://github.com/ultralytics/assets/releases