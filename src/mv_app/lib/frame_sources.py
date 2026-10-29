
import cv2
import threading
from queue import Queue, Full
from abc import ABC, abstractmethod
from pathlib import Path
import logging
log = logging.getLogger(__name__)

import numpy as np
from pyservicemaker import Pipeline, BufferRetriever, Receiver
import torch
import time

# import gi
# gi.require_version('Gst', '1.0')
# from gi.repository import Gst, GLib # pyright: ignore
# Gst.init(None)

from .utils import (
    ValidationError,
)
from .frame import Frame

class FrameSource(ABC):

    @abstractmethod
    def connect(self, *args, **kwargs): pass

    @abstractmethod
    def disconnect(self): pass

    @abstractmethod
    def get_frame(self, cuda_tensor=True) -> Frame: pass

    @property
    @abstractmethod
    def fps(self) -> float: pass

    @property
    @abstractmethod
    def frame_count(self) -> int: pass
    
    @property
    @abstractmethod
    def frame_width(self) -> int: pass

    @property
    @abstractmethod
    def frame_height(self) -> int: pass

class UVCCamCv2(FrameSource):

    def __init__(self, device_paths:list[str]):

        self._device_paths = device_paths
        self._video_caps:list[cv2.VideoCapture] = []

        self._fps = 0
        self._frame_width = 0
        self._frame_height = 0
        self._frame_count = 0
        self._total_frames = 0

    def connect(self, buffer_size:int=1, frame_size:tuple[int, int]=(), fps:int=0):

        for device_path in self._device_paths:
            if not Path(device_path).exists(): 
                raise ValidationError(f"Video device '{device_path}' does not exist!")
        
        for device_path in self._device_paths:
            self._video_caps.append(cv2.VideoCapture(device_path, cv2.CAP_V4L2))
            if not self._video_caps[-1].isOpened():
                self.disconnect() # disconnect from all video caps
                raise ValidationError(f"Cannot open video device: '{device_path}'")
        
        for video_cap in self._video_caps:
            video_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
            video_cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

            if frame_size:
                video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[0])
                video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[1])
            if fps:
                video_cap.set(cv2.CAP_PROP_FPS, fps)

        # validate that all sources have the same image shape and fps
        heights = []
        widths = []
        fpss =[]
        for video_cap in self._video_caps:
            heights.append(int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            widths.append(int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            fpss.append(int(video_cap.get(cv2.CAP_PROP_FPS)))
        
        self._frame_height = heights[0]
        if len(set(heights)) > 1:
            raise ValidationError("Frame height must be the same for all sources")
        
        self._frame_width = widths[0]
        if len(set(widths)) > 1:
            raise ValidationError("Frame widths must be the same for all sources")
        
        self._fps = fpss[0]
        if len(set(fpss)) > 1:
            raise ValidationError("FPS must be the same for all sources")

        self._total_frames = np.inf
        self._frame_count = 0

        video_specs = f"{self._frame_width}x{self._frame_height}@{self._fps}"
        log.info(f"cv2 connected to: {self._device_paths} - WxH@fps = {video_specs}")
    
    def disconnect(self):

        for video_cap in self._video_caps:
            if video_cap:
                video_cap.release()
        self._video_caps = []
        
        self._fps = 0.0
        self._frame_width = 0
        self._frame_height = 0
        self._frame_count = 0
        self._total_frames = 0
        log.info(f"disconnected from video devices: {self._device_paths}")

    def get_frame(self, cuda_tensor=True) -> Frame:
        '''
        Returns a Frame with the image in RGB pixel format
        '''

        frame = Frame(
            [], 
            self._frame_count, 
            self._total_frames, 
            self._fps
        )
        with frame.time_it("grab_frame"):
            # grab the raw images from cameras
            for video_cap in  self._video_caps:
                video_cap.grab()

            # then retrieve them, which is slower but now they are more in sync
            tensor_list = []
            for device_path, video_cap in zip(self._device_paths, self._video_caps):
                success, image = video_cap.retrieve()
                if success:
                    # upload array to cuda tensor as HWC
                    device = 'cuda' if cuda_tensor else 'cpu'
                    image_tensor = torch.as_tensor(image, dtype=torch.uint8, device=device)
                    
                    # reformat BGR to RGB and keep HWC layout
                    image_tensor = image_tensor[:, :, [2, 1, 0]]
                    
                    tensor_list.append(image_tensor)
                else:
                    tensor_list.append(None)
                if not success or image is None:
                    log.info(f"Could not get image from device {device_path}. Reached end of video.")
            
        frame.image_tensors = tensor_list
        self._frame_count += 1
        return frame

    @property
    def fps(self):
        return self._fps

    @property
    def total_frames(self):
        return self._total_frames
    
    @property
    def frame_count(self):
        return self._frame_count

    @property
    def frame_width(self):
        return self._frame_width
    
    @property
    def frame_height(self):
        return self._frame_height

class UVCCamGst(FrameSource):

    def __init__(self, device_paths:list[str]):

        self._device_paths = device_paths

        self._fps = 0
        self._frame_width = 0
        self._frame_height = 0
        self._frame_count = 0
        self._total_frames = 0

        self._frame_retriever = None
        self._pipeline:Pipeline = None # pyright: ignore
        self._grab_thread:threading.Thread = None # pyright: ignore
        self._buffer_maxsize = 1
        self._buffer:Queue = None # pyright: ignore

    def _get_pipeline(self, appsink_name:str):

        pipeline = Pipeline("stereo-webcams")

        pipeline.add("nvstreammux", "mux", { # sink and src only produce video/x-raw(memory:NVMM)
            "width": self._frame_width,
            "height": self._frame_height,
            "batch-size": len(self._device_paths),
            "batched-push-timeout": int(1e6 / (self._fps)), # time in microsecs to wait for both buffers
            "live-source": True,
            "max-latency": int(1e9 / (self._fps)), # in live mode, extra delay in nanosecs to allow buffers to create
            "sort-batch": True, # required bc we used sink_%u and pads are auto-assigned images
            # "align-inputs": True, # added
            # "sync-inputs": True, # added
        })
    
        pipeline.add("nvvideoconvert", "convert", {
            "copy-hw": 1, # (1): GPU (def), (2): VIC 
            "compute-hw": 1,  # (1): GPU, (2): VIC - VIC does not support RBG conversions
            "nvbuf-memory-type": 4 # (4): nvbuf-mem-surface-array, (2): nvbuf-mem-cuda-device
        })

        pipeline.add("capsfilter", "output_caps", {
            "caps": "video/x-raw(memory:NVMM),format=RGB",
        })
    
        pipeline.add("appsink", appsink_name, {
            "emit-signals": True,
            "drop": True,
            "max-buffers": 1,
            "sync": False, #Set sync=True for real-time extraction
        })
        pipeline.link("mux", "convert", "output_caps", appsink_name)
        
        
        for idx, device in enumerate(self._device_paths):

            cam_source_name = f"camera_{idx}"
            pipeline.add("v4l2src", cam_source_name, {
                "device": device,
                "do-timestamp": True,
                "io-mode": 4,  # io-mode=4=dmabuf, 2=mmap would work too
            })

            cam_caps_name = f"input_caps_{idx}"
            pipeline.add("capsfilter", cam_caps_name, {
                "caps": f"image/jpeg,width={self._frame_width},height={self._frame_height},framerate={self._fps}/1",
            })

            decoder_name = f"decoder_{idx}"
            pipeline.add("nvv4l2decoder", decoder_name, { # only src cap is video/x-raw(memory:NVMM)
                "mjpeg": True,
            })

            queue_name = f"queue_{idx}"
            pipeline.add("queue", queue_name, {
                "max-size-buffers": 1,
                "leaky": 2,  # leaky=downstream, discards old frames when full
            })
            pipeline.link(cam_source_name, cam_caps_name, decoder_name, queue_name)
    
            # the sink_%u notation is required. it does work with sink_0, but it is not recommended
            # https://github.com/NVIDIA/skills/blob/main/skills/deepstream-dev/references/service_maker_api.md
            # Link the queue src pad to a mux sink pad
            pipeline.link((queue_name, "mux"), ("", "sink_%u"))
    
        return pipeline

    def _grab_frames_thread_func(self):

        class FrameRetriever(BufferRetriever):

            def __init__(self, batch_size:int, buffer_put_timeout:float, buffer:Queue):
                super().__init__()
        
                self.batch_size = batch_size
                self.buffer_put_timeout = buffer_put_timeout
                self.buffer = buffer

                self.t0 = time.perf_counter()
                self.speed_ms = -1
        
            def consume(self, buffer_retriever):

                t_now = time.perf_counter()
                self.speed_ms = (t_now - self.t0) * 1000
                self.t0 = t_now
                
                # the resulting tensors are in HWC layout
                tensor_list = [None] * self.batch_size
                for frame_meta in buffer_retriever.batch_meta.frame_items:
                    tensor = buffer_retriever.extract(frame_meta.source_id).clone()
                    image_tensor = torch.utils.dlpack.from_dlpack(tensor) # pyright: ignore
                    tensor_list[frame_meta.source_id] = image_tensor

                # if tensor_list[0] is None:
                #     log.warning("Tensor list 0 is missing")
                # if self.batch_size > 1 and tensor_list[1] is None:
                #     log.warning("Tensor list 1 is missing")

                try:
                    if all(tensor is not None for tensor in tensor_list):
                        self.buffer.put_nowait(tensor_list)
                    # self.buffer.put(tensor_list, timeout=self.buffer_put_timeout)
                except Full:
                    try: self.buffer.get_nowait()
                    except: pass

                    try: self.buffer.put_nowait(tensor_list)
                    except: pass
                except: pass

                return 1

        sink_name = "imagesink"
        self._pipeline = self._get_pipeline(sink_name)

        buffer_put_timeout = 1 / self._fps
        retriever = FrameRetriever(len(self._device_paths), buffer_put_timeout, self._buffer)
        self._frame_retriever = retriever
        
        self._pipeline.attach(
            sink_name, 
            Receiver("receiver", retriever), 
            tips="new-sample"
        )

        try:
            video_specs = f"{self._frame_width}x{self._frame_height}@{self._fps}"
            log.info(f"Gst connected to : {self._device_paths} - WxH@fps = {video_specs}")
            self._pipeline.start()
            self._pipeline.wait()

        except Exception as ex:
            log.error(f"Gst grab error: {ex}")

        finally:
            self._pipeline.stop()
            self._buffer = None
            log.info(f"Gst disconnected from video devices: {self._device_paths}")
    
    def connect(self, buffer_size:int=1, frame_size:tuple[int, int]=(0,0), fps:int=0):

        for device_path in self._device_paths:
            if not Path(device_path).exists(): 
                raise ValidationError(f"Video device '{device_path}' does not exist!")
        
        self._fps = int(fps) or 30
        self._frame_height = int(frame_size[0]) or 480
        self._frame_width = int(frame_size[1]) or 640
        
        self._total_frames = np.inf
        self._frame_count = 0

        # buffer is created here and not in the thread, because the thread might take too long to start
        # and the first get_frame will find an uninitialized buffer.
        self._buffer_maxsize = buffer_size
        self._buffer = Queue(maxsize=self._buffer_maxsize)
        self._grab_thread = threading.Thread(target=self._grab_frames_thread_func)
        self._grab_thread.daemon = True
        self._grab_thread.start()
    
    def disconnect(self):

        self._pipeline.stop()
        self._fps = 0.0
        self._frame_width = 0
        self._frame_height = 0
        self._frame_count = 0
        self._total_frames = 0
    
    def get_frame(self, cuda_tensor=True) -> Frame:
        '''
        Returns a Frame with the image in RGB pixel format
        '''
        
        frame = Frame(
            [], 
            self._frame_count, 
            self._total_frames, 
            self._fps
        )

        # the speed of the gst thread frame production must not be included in any metrics
        # as this is an async process that does not account for the overall processing rate
        if self._frame_retriever:
            frame.perf['_gst'] = self._frame_retriever.speed_ms

        with frame.time_it("grab"):
            try:
                tensor_list:list[torch.Tensor] = self._buffer.get()
                
                if not cuda_tensor:
                    tensor_list = [tensor.cpu() for tensor in tensor_list]

            except Exception as ex:
                tensor_list = []
                log.warning(f"Could not get tensors: {ex}")

        frame.image_tensors = tensor_list

        self._frame_count += 1
        return frame

    @property
    def fps(self):
        return self._fps

    @property
    def total_frames(self):
        return self._total_frames
    
    @property
    def frame_count(self):
        return self._frame_count

    @property
    def frame_width(self):
        return self._frame_width
    
    @property
    def frame_height(self):
        return self._frame_height

class VideoFile(FrameSource):

    def __init__(self, file_paths:list[str]):

        self._file_paths = file_paths
        self._video_caps:list[cv2.VideoCapture] = []
        self._is_sbs = False

        self._fps = 0.0
        self._frame_width = 0
        self._frame_height = 0
        self._frame_count = 0
        self._total_frames = 0

    def connect(self, **kwargs):

        # if the input filepaths has 2 values and they are exactly the same, then an SBS video is requested
        self._is_sbs = len(self._file_paths) == 2 and len(set(self._file_paths)) == 1
        if self._is_sbs:
            self._file_paths.pop(1)

        for file_path in self._file_paths:
            if not Path(file_path).exists(): 
                raise ValidationError(f"Video '{file_path}' does not exist!")
        
        for file_path in self._file_paths:
            self._video_caps.append(cv2.VideoCapture(file_path))
            if not self._video_caps[-1].isOpened():
                self.disconnect() # disconnect from all video caps
                raise ValidationError(f"Cannot open video: '{file_path}'")
        
        # validate that all sources have the same image shape and fps
        heights = []
        widths = []
        fpss =[]
        for video_cap in self._video_caps:
            heights.append(int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            widths.append(int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            fpss.append(video_cap.get(cv2.CAP_PROP_FPS))
        
        self._frame_height = heights[0]
        if len(set(heights)) > 1:
            raise ValidationError("Frame height must be the same for all sources")
        
        self._frame_width = widths[0]
        if len(set(widths)) > 1:
            raise ValidationError("Frame widths must be the same for all sources")
        
        self._fps = fpss[0]
        if len(set(fpss)) > 1:
            raise ValidationError("FPS must be the same for all sources")

        self._total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_count = 0

        log.info(f"connected to video: {self._file_paths} - as SBS: {self._is_sbs}")
    
    def disconnect(self):

        for video_cap in self._video_caps:
            if video_cap:
                video_cap.release()
        self._video_caps = []
        
        self._fps = 0.0
        self._frame_width = 0
        self._frame_height = 0
        self._frame_count = 0
        self._total_frames = 0
        log.info(f"disconnected from video devices: {self._file_paths}")

    def get_frame(self) -> Frame:
        '''
        Returns a Frame with the image in RGB pixel format
        '''

        images = []
        for file_path, video_cap in zip(self._file_paths, self._video_caps):
            success, image = video_cap.read()
            if success:
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)
                # if image is a side by side frame, then there should by a single file source
                # so we can take the image, split it and break
                if self._is_sbs:
                    images = np.hsplit(image, 2)
                    break
                images.append(image)
            else:
                log.info(f"Could not get image from file {file_path}. Reached end of video.")
                images.append(None)
            
            
        frame = Frame(
            images, 
            self._frame_count, 
            self._total_frames, 
            self._fps
        )
                
        self._frame_count += 1
        return frame

    @property
    def fps(self):
        return self._fps

    @property
    def total_frames(self):
        return self._total_frames
    
    @property
    def frame_count(self):
        return self._frame_count

    @property
    def frame_width(self):
        return self._frame_width
    
    @property
    def frame_height(self):
        return self._frame_height

