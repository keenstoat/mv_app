
# Object Detection with Stereo Depth Estimation</h1>

**Real-time object detection and stereo depth estimation using NVIDIA Jetson Orin Nano, TensorRT, DeepStream, CV-CUDA, and Python.**

A GPU-accelerated computer vision application that combines YOLO object detection with stereo camera geometry to estimate the distance of detected objects from a binocular camera setup.

The application provides an interactive web interface for configuring video sources, selecting inference modes, enabling stereo depth analysis, and visualizing detection results in real time.

![Application Demo](docs/demo-03.gif)


## Overview

This project explores the implementation of an end-to-end computer vision pipeline on an NVIDIA Jetson Orin Nano, integrating object detection, stereo image processing, and real-time visualization.

Two synchronized video streams are captured from a binocular webcamera setup and processed through a configurable pipeline. Depending on the selected mode, the application can perform:

- **Object detection:** Identify and classify objects in both camera streams using a YOLO model accelerated with TensorRT.
- **Object detection with depth estimation:** Rectify stereo images, detect corresponding objects, and estimate their distance using triangulation.
- **Dense stereo depth estimation:** Generate disparity and depth maps from rectified stereo images.

The application is built with **NiceGUI**, providing a browser-based interface to configure the pipeline and visualize results without requiring a dedicated desktop application.

A primary design objective was to **minimize CPU-GPU data transfers** by keeping image processing operations on the GPU wherever practical.

## Key Features

- Stereo video capture from two USB webcams or side-by-side (SBS) video files.
- GPU-based image processing using PyTorch and CV-CUDA CUDA tensors.
- Stereo image rectification accelerated with CV-CUDA.
- Object detection using TensorRT enabled YOLO models.
- Two inference execution modes: sequential and optimized with overlapping GPU operations.
- Object correspondence between stereo views using ByteTrack multiple object tracking.
- Sparse stereo depth estimation using geometric triangulation.
- Dense disparity and depth map computation.
- Browser-based configuration and visualization with NiceGUI.
- Custom WebSocket streaming with synchronized image and detection metadata.
- Alternative WebRTC visualization.

## Hardware Setup

The system runs on an [NVIDIA Jetson Orin Nano - Developer Kit](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/index.html), connected to two identical [USB webcams](https://emeet.com/products/webcam-c950?variant=42290847187115).

The webcams are mounted side by side on a flat base to form a binocular stereo camera arrangement.

## Software Stack

The following software platforms and tools were used.

| Component         | Version    |
|-------------------|------------|
| NVIDIA JetPack    | 7.2.1      |
| NVIDIA L4T        | 39.2.1     |
| Ubuntu            | 24.04      |
| CUDA              | 13.2.86    |
| TensorRT          | 10.16.2.10 |
| NVIDIA VPI        | 4.1.4.0    |
| NVIDIA DeepStream | 9.2.1      |
| Python            | 3.12.3     |

Besides DeepStream, all other tools come included with `Jetpack 7.2.1` and the `nvidia-jetpack` package.

### Main Libraries and Frameworks

- **NiceGUI:** Web application and interactive user interface.
- **DeepStream / PyServiceMaker:** UVC video capture, synchronization, and GStreamer pipeline integration.
- **PyTorch:** CUDA tensor representation and GPU data handling.
- **CV-CUDA:** GPU-accelerated image processing.
- **TensorRT:** Optimized YOLO inference for NVIDIA devices.
- **OpenCV:** General image processing.
- **ByteTrack:** Association of detected objects in binocular frames.
- **WebSocket / WebRTC:** Real-time streaming for visualization.

## Application Interface

The application provides a web interface organized into two main areas.

### Configuration Panel

The left panel contains tabbed sections to select video sources and vision models, set configuration and processing options, and view runtime data.

| | | |
| :---: | :---: | :---: |
| ![](docs/explorer-view.jpg) | ![](docs/config-view.jpg) | ![](docs/runtime-view.jpg) |
|explorer view | configuration view | runtime view|

Users can:

- Select the left and right video sources from available USB cameras or SBS video files.
- Load and uload YOLO object detection models.
- Select the webcam image size and frame rate.
- Select between sequential and optimized inference for object detection.
- Enable stereo triangulation for depth estimation.
- Configure the available processing and visualization options.

### Visualization Panel

The right panel displays the processed video streams and their associated results in real-time.

![](docs/app-interface.jpg)

## Processing Pipeline

The application uses a modular processing pipeline, allowing different computer vision operations to be enabled depending on the selected mode.

The main stages are:

```mermaid
graph TD
  A(image capture) --> B(rectification)
  B(rectification) --> C(object detection)
  C(object detection) --> D(depth estimation)
  D(depth estimation) --> E(Visualization)
```

### Stage 1: Image Capture and Synchronization

Image capture is implemented using [NVIDIA DeepStream PyServiceMaker](https://docs.nvidia.com/deepstream/text/DS_service_maker_intro#3-python-pipeline-api), which provides a Python interface for constructing and managing a GStreamer-based processing pipeline.

The following is the GStreamer capture pipeline. It consumes frames from both webcams, ingests them into the GPU space as NvBufSurface surfaces, and synchronizes them: 

```
nvstreammux name=mux 
    width={width} height={height} 
    batch-size=2 batched-push-timeout={1e6 / fps} 
    live-source=true max-latency={1e9 / fps}
    sort-batch=true
! nvvideoconvert 
    copy-hw=GPU compute-hw=GPU nvbuf-memory-type=nvbuf-mem-surface-array 
! video/x-raw(memory:NVMM), format=RGB 
! appsink 
    name=app_sink emit-signals=true drop=true max-buffers=1 sync=false

v4l2src 
    device=/device/video0 do-timestamp=true io-mode=dmabuf 
! image/jpeg, width={width}, height={height}, framerate={fps}/1 
! nvv4l2decoder mjpeg=true 
! queue max-size-buffers=1 leaky=downstream 
! mux.sink_%u

v4l2src 
    device=/device/video1 do-timestamp=true io-mode=dmabuf 
! image/jpeg, width={width}, height={height}, framerate={fps}/1 
! nvv4l2decoder mjpeg=true 
! queue max-size-buffers=1 leaky=downstream 
! mux.sink_%u
```

Finally, a [BufferRetriever](https://github.com/NVIDIA/skills/blob/main/skills/deepstream-dev/references/buffer_apis.md) object is attached to the pipeline's appsink to  allow the consumption of the pipeline output images as CUDA-backed PyTorch tensors.

```python
class FrameRetriever(BufferRetriever):
    ...
    def consume(self, buffer_retriever):
        tensor = buffer_retriever.extract(frame_meta.source_id).clone()
        image_tensor:torch.Tensor = torch.utils.dlpack.from_dlpack(tensor)
```


### Stage 2: Stereo Image Rectification

> **This stage is only required when performing stereo depth estimation.**

For stereo depth estimation, the captured images must be rectified to compensate for camera distortion and align their epipolar geometry, i-e., ensuring that any point in the left image maps to a point on the exact same row in the right image, reducing the stereo matching search space to a 1D horizontal path.

Rectification maps were generated beforehand using OpenCV calibration and stereo geometry functions. See SECTION XXX for more info.

Image rectification is done using [CV-CUDA's remap function](https://cvcuda.github.io/CV-CUDA/modules/python/operators.html#cvcuda.remap), allowing the operation to be applied directly on the CUDA-backed PyTorch tensors.

### Stage 3: Object Detection

Object detection is implemented using a [YOLO26 end-to-end model](https://docs.ultralytics.com/guides/end2end-detection) exported to a **TensorRT engine**.

Although Ultralytics is used to export the model, the runtime inference implementation does not depend on Ultralytics. Instead, custom preprocessing, inference, and postprocessing functions were developed allowing the operation to be applied directly on the CUDA-backed PyTorch tensors.

#### Preprocessing

Because the input image tensors come from actual images, they have a HWC layout. These tensors are stacked (as NHWC layout) and processed using [CV-CUDA resize_crop_convert_reformat function](https://cvcuda.github.io/CV-CUDA/modules/python/operators.html#cvcuda.resize_crop_convert_reformat) which performs all the required pre-processing transformations resulting in still a CUDA-backed PyTorch tensor with NCHW layout, as required by YOLO.

#### Inference

The inference supports two modes:

**Sequential inference**: Preprocessing, inference, and postprocessing operations are executed in sequence.

**Optimized inference**: A pipelined execution strategy uses two CUDA streams to overlap preprocessing of an upcoming image with inference on another image. This approach is intended to improve GPU utilization and reduce overall processing latency.

#### Postprocessing

Finally, the postprocessing extracts and transforms the detections resulting in numpy arrays, which are CPU-backed. 

Each detction includes:
- `xyxy`: Bounding box coordinates
- `conf`: Detection confidence
- `class_id`: Predicted object class

### Stage 4: Depth Estimation

Once objects have been detected in both rectified images, the application estimates their distance from the cameras. The process consists of two main operations.

#### Stereo Object Association

Detections from the left and right images are associated using [ByteTrack](https://trackers.roboflow.com/develop/guides/track/#trackers) multiple object tracker, in order to identify detections corresponding to the same physical object in both camera views.

#### Geometric Triangulation

For each associated pair of detections, the center coordinates of their bounding boxes are extracted.

These coordinates are triangulated using [OpenCV's triangulatePoints()](https://docs.opencv.org/4.13.0/d0/dbd/group__triangulation.html) function. 

The projection matrices are obtained from the stereo rectification process. See section XXXX for more info.

Triangulation reconstructs a 3D point from the corresponding image coordinates and the calibrated camera geometry. The estimated distance from an object to the camera is then calculated as the euclidean distance, i.e. the distance in a straight line.


### Stage 5: Visualization and Streaming

The final stage exposes the images and their associated detection metadata for visual presentation over WebRTC or a WebSocket with a custom data stream.

#### WebRTC Streaming

This visualization method exposes annotated images through a WebRTC `MediaStreamTrack`.

This approach produces smoother video playback, but the annotations rendered directly into the video frames show pixelation due to video encoding.

#### Custom WebSocket Streaming

This visualization mechanism uses a custom WebSocket-based streaming protocol.

Each transmitted message bundles the image and its corresponding detection metadata into a single byte stream.

On the client side, the message is decoded and the image is rendered together with its annotations.

These annotations include:

- Detection bounding boxes
- Object class labels
- Estimated object distance in centimeters

Bundling image and metadata together ensures that annotations remain associated with the correct image, avoiding synchronization problems that can arise when video and detection results are transmitted independently.



## Camera Stereo Calibration

Accurate stereo depth estimation requires knowledge of the intrinsic and extrinsic camera parameters.

The calibration procedure determines camera intrinsics, lens distortion coefficients, the relative camera pose, and the projection matrices used for triangulation.

The resulting parameters are also used to generate the rectification maps applied during runtime.

**[TODO: Add complete camera calibration procedure, sample calibration images, calibration results, and configuration file format.]**

https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html

## Model Export and TensorRT Optimization

The object detection model is exported from its original PyTorch representation to a TensorRT engine using the Ultralytics export functionality.

The application then loads the generated engine and performs inference through a custom TensorRT execution implementation.

This separation allows the model preparation workflow to use Ultralytics while retaining direct control over preprocessing, inference execution, and postprocessing during runtime.

**[TODO: Add export commands, model configuration, supported input shapes, precision settings, and optimization details.]**

## Performance Evaluation

Performance evaluation is an important part of this project, particularly because the implementation targets an embedded GPU platform.

The following measurements will be documented:

| Metric                                | Result ms     |
|---------------------------------------|---------------|
| Camera capture and synchronization    | 33.46 ± 3.31  |
| Frame grab                            | 0.02 ± 0.01   |
| Stereo rectification latency          | 2.22 ± 0.78   |
| Sequential inference latency          | 23.08 ± 2.98  |
| Optimized inference latency           | 16.58 ± 5.44  |
| Object association and triangulation  | 0.5 ± 0.44    |
| End-to-end processing latency         | 25.83         |
| Visualization frame rate              | 25.02 fps     |

The evaluation will also compare sequential and optimized inference execution to quantify the benefits of overlapping GPU operations.

**[TODO: Add benchmarks, methodology, GPU utilization, and execution timelines.]**

## Technical Design Considerations

Several implementation decisions shaped the architecture of this project.

**GPU-resident image processing**

Images are represented as PyTorch CUDA tensors throughout the main processing stages, minimizing transfers between CPU and GPU memory.

**Direct TensorRT execution**

Custom inference functions avoid unnecessary framework overhead and provide control over CUDA execution and memory operations.

**Asynchronous GPU processing**

Separate CUDA streams enable overlapping operations and provide an opportunity to improve processing throughput.

**Modular processing pipeline**

The application separates acquisition, rectification, detection, depth estimation, and visualization into distinct stages.

**Synchronized visualization metadata**

The WebSocket streaming protocol transmits images and detection results together to preserve their correspondence.

**Selective CPU processing**

Operations involving relatively small amounts of metadata, such as object association and triangulation, are executed on the CPU when the overhead of GPU execution is not justified.

## Limitations and Future Improvements

Current limitations and potential areas for improvement include:

- **Depth accuracy:** Bounding box centers are only approximations of corresponding physical points.
- **Object correspondence:** Tracking-based association may produce incorrect matches, particularly when multiple similar objects are visible.
- **Stereo camera synchronization:** Two independent USB webcams do not provide hardware-triggered synchronization.
- **Calibration stability:** The camera mounting arrangement must remain mechanically stable to preserve calibration accuracy.
- **Visualization tradeoffs:** WebSocket and WebRTC approaches have different smoothness, encoding, and annotation-quality characteristics.

Potential future improvements include evaluating depth accuracy against measured ground-truth distances, improving stereo object correspondence, and further optimizing the GPU processing pipeline.

## Installation and Usage

**[TODO: Add prerequisites, dependencies, installation instructions, model preparation, camera calibration requirements, and application startup commands.]**

## Project Structure

**[TODO: Add repository directory tree and explain the main modules.]**

## References and Acknowledgments

**[TODO: Add documentation links, referenced libraries, algorithms, and relevant technical resources.]**

## License

**[TODO: Specify the project license.]**