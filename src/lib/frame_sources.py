import time
import numpy as np
import cv2
import threading
from queue import Queue, Full, Empty
from abc import ABC, abstractmethod
from pathlib import Path
import logging as log

import gi
import numpy as np
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
Gst.init(None)

from .utils import (
    ValidationError, MsgType,
)
from .frame import Frame

class FrameSource(ABC):

    @abstractmethod
    def connect(self, *args, **kwargs): pass

    @abstractmethod
    def disconnect(self): pass

    @abstractmethod
    def source_name(self) -> str: pass

    @abstractmethod
    def get_frame(self) -> Frame: pass

    @property
    @abstractmethod
    def fps(self) -> float: pass

    @property
    @abstractmethod
    def frame_count(self) -> int: pass
    
    @property
    @abstractmethod
    def frame_count(self) -> int: pass
    
    @property
    @abstractmethod
    def frame_width(self) -> int: pass

    @property
    @abstractmethod
    def frame_height(self) -> int: pass

class UVCCam(FrameSource):

    def __init__(self, device_paths:dict[int, str]):

        self._device_paths = device_paths
        self._video_caps:dict[int, cv2.VideoCapture] = {}

        self._fps = 0.0
        self._frame_width = 0
        self._frame_height = 0
        self._frame_count = 0
        self._total_frames = 0

    def connect(self, buffer_size:int=1, frame_size:tuple[int, int]=(), fps:float=0.0):

        for device_path in self._device_paths.values():
            if not Path(device_path).exists(): 
                raise ValidationError(f"Video device '{device_path}' does not exist!")
        
        for side, device_path in self._device_paths.items():
            self._video_caps[side] = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
            if not self._video_caps[side].isOpened():
                self.disconnect() # disconnect from all video caps
                raise ValidationError(f"Cannot open video device: '{device_path}'")
        
        for side, video_cap in self._video_caps.items():
            video_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
            video_cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

            if frame_size:
                video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[0])
                video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[1])
                log.info(f"Device pos {side} frame shape set to HxW: {frame_size} ")
            if fps:
                video_cap.set(cv2.CAP_PROP_FPS, fps)
                log.info(f"Device pos {side} FPS set to {fps} ")

        # validate that all sources have the same image shape and fps
        heights = []
        widths = []
        fpss =[]
        for video_cap in self._video_caps.values():
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

        self._total_frames = np.inf
        self._frame_count = 0

        log.info(f"connected to video devices: {self._device_paths}")
    
    def disconnect(self):

        for video_cap in self._video_caps.values():
            if video_cap:
                video_cap.release()
        self._video_caps = {}
        
        self._fps = 0.0
        self._frame_width = 0
        self._frame_height = 0
        self._frame_count = 0
        self._total_frames = 0
        log.info(f"disconnected from video devices: {self._device_paths}")

    def source_name(self, side:int=None):
        return self._device_paths if side is None else self._device_paths[side]

    def get_frame(self) -> Frame:
        '''
        Returns a Frame with the image in RGB pixel format
        '''

        # grab the raw images from cameras
        for video_cap in  self._video_caps.values():
            video_cap.grab()

        # then retrieve them, which is slower but now they are more in sync
        images = []
        for side, video_cap in self._video_caps.items():
            success, image = video_cap.retrieve()
            if success:
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)

            images.append(image)
            if not success or image is None:
                device_path = self._device_paths[side]
                log.info(f"Could not get image from device {device_path}. Reached end of video.")
        
        if len(images) == 1:
            image = images[0]  
        elif np.all(img is not None for img in images):
            image = np.hstack(images)
        else:
            image = None
        frame = Frame(
            image, 
            len(images), 
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

class UVCCamGst(FrameSource):

    def __init__(self, device_paths:dict[int, str]):

        self._device_paths = device_paths

        self._fps = 0.0
        self._frame_width = 0
        self._frame_height = 0
        self._frame_count = 0
        self._total_frames = 0

        self._grab_thread:threading.Thread = None
        self._buffer_maxsize = 1
        self._buffer:Queue = None
        self._gst_loop = None
        self._use_gst_frame_nvmm = False

    def _get_pipeline_str(self, appsink_name):

        w = self._frame_width
        h = self._frame_height
        fps = self._fps
        devices = self._device_paths
        mem = "(memory:NVMM)" if self._use_gst_frame_nvmm else ""

        if len(devices) == 1:
            #"nvjpegdec" should be faster than "nvv4l2decoder mjpeg=true"  but cant make the former work
            return (
                f"v4l2src device={list(devices.values())[0]} do-timestamp=true io-mode=dmabuf ! image/jpeg, width={w}, height={h}, framerate={fps}/1 ! "
                "nvv4l2decoder mjpeg=true ! "
                f"nvvideoconvert compute-hw=GPU ! video/x-raw{mem}, format=RGB ! "
                f"appsink name={appsink_name} emit-signals=true drop=true max-buffers=1 sync=true "
            )
            
        if len(devices) == 2:
            # sync-inputs=true align-inputs=true
            return (
                f"nvstreammux name=mux width={w} height={h} batch-size=2 batched-push-timeout={int(1e6 / (fps * 2))} live-source=true max-latency={int(1e6 / (fps * 2.5))} ! "
                f"nvmultistreamtiler rows=1 columns=2 width={w * 2} height={h} ! "
                f"nvvideoconvert compute-hw=GPU ! video/x-raw{mem}, format=RGB ! "
                f"appsink name={appsink_name} emit-signals=true drop=true max-buffers=1 sync=false "
                
                f"v4l2src device={devices[0]} do-timestamp=true io-mode=dmabuf ! image/jpeg, width={w}, height={h}, framerate={fps}/1 ! "
                "nvv4l2decoder mjpeg=true ! queue max-size-buffers=1 leaky=downstream ! mux.sink_0 "
                
                f"v4l2src device={devices[1]} do-timestamp=true io-mode=dmabuf ! image/jpeg, width={w}, height={h}, framerate={fps}/1 ! "
                "nvv4l2decoder mjpeg=true ! queue max-size-buffers=1 leaky=downstream ! mux.sink_1"
            )
        
    def _grab_frames_thread_func(self):

        def on_new_sample(sink):

            sample = sink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.ERROR
                
            buff = sample.get_buffer()
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            
            total_width = structure.get_value("width")
            height = structure.get_value("height")

            success, map_info = buff.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.OK

            try:
                image = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, total_width, 3))
                try:
                    self._buffer.put(image, timeout=buffer_put_timeout)
                except Full:
                    try: self._buffer.get_nowait()
                    except Empty: pass

                    try: self._buffer.put_nowait(image)
                    except Full: pass
                
            finally:
                buff.unmap(map_info)
            
            return Gst.FlowReturn.OK

        buffer_put_timeout = 1 / self._fps
        sink_name = "imagesink"
        pipeline_string = self._get_pipeline_str(sink_name)
        # log.info(f"gst pipeline: {pipeline_string}")
        pipeline = Gst.parse_launch(pipeline_string)
        appsink = pipeline.get_by_name(sink_name)
        appsink.connect("new-sample", on_new_sample)
        pipeline.set_state(Gst.State.PLAYING)
        
        self._gst_loop = GLib.MainLoop()
        try:
            log.info(f"Gst connected to video devices: {self._device_paths}")
            self._gst_loop.run()

        except Exception as ex:
            log.error(f"Gst loop error: {ex}")

        finally:
            self._buffer = None
            pipeline.set_state(Gst.State.NULL)
            log.info(f"Gst disconnected from video devices: {self._device_paths}")

    def connect(self, buffer_size:int=1, frame_size:tuple[int, int]=(0,0), fps:float=0.0):

        for device_path in self._device_paths.values():
            if not Path(device_path).exists(): 
                raise ValidationError(f"Video device '{device_path}' does not exist!")
        
        self._fps = fps or 30
        self._frame_height = frame_size[0] or 480
        self._frame_width = frame_size[1] or 640
        
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

        GLib.idle_add(self._gst_loop.quit)
        self._fps = 0.0
        self._frame_width = 0
        self._frame_height = 0
        self._frame_count = 0
        self._total_frames = 0
    
    def source_name(self, side:int=None):
        return self._device_paths if side is None else self._device_paths[side]

    def get_frame(self) -> Frame:
        '''
        Returns a Frame with the image in RGB pixel format
        '''
        
        try:
            image = self._buffer.get() # waits indefinitely for a new image
        except:
            image = None

        frame = Frame(
            image, 
            len(self._device_paths), 
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

class UVCCamFFmpeg(FrameSource):

    def __init__(self, device_paths:dict[int, str]):

        self._device_paths = device_paths

        self._fps = 0.0
        self._frame_width = 0
        self._frame_height = 0
        self._frame_count = 0
        self._total_frames = 0

        self._grab_thread:threading.Thread = None
        self._buffer_maxsize = 1
        self._buffer:Queue = None
        self._thread_active = False
   
    def _grab_frames_thread_func(self):
        import ffmpeg
        input_opts = {
            'f': 'v4l2', 
            'input_format': 'mjpeg',
            's': f'{self._frame_width}x{self._frame_height}', 
            'framerate': str(self._fps),
            'thread_queue_size': 1,
            'use_wallclock_as_timestamps': 1,

        }
        
        device_paths = list(self._device_paths.values())
        if len(device_paths) == 1:
            video = (
                ffmpeg.input(device_paths[0], **input_opts)
                .filter("setpts", "PTS-STARTPTS")
            )
        else:
            cams = []
            for device_path in device_paths:
                cam = (
                    ffmpeg.input(device_path, **input_opts)
                    .filter("setpts", "PTS-STARTPTS")
                )
                cams.append(cam)
            video = ffmpeg.filter(cams, "hstack")
            
            
        out = video.output(
            'pipe:', 
            format='rawvideo', 
            pix_fmt='rgb24',
            fflags='nobuffer',
            flags='low_delay',
        )

        # out = out.global_args('-loglevel', 'quiet')

        process = out.run_async(pipe_stdout=True)
        log.info(f"ffmpeg connected to video devices: {self._device_paths}")

        width = self._frame_width * len(device_paths)
        height = self._frame_height
        channels = 3
        frame_bytes_len = height * width * channels

        try:
            self._thread_active = True
            
            buffer_put_timeout = 1/self._fps
            while self._thread_active:
                image_bytes = process.stdout.read(frame_bytes_len)
                if not image_bytes or len(image_bytes) != frame_bytes_len:
                    break
                
                image = np.frombuffer(image_bytes, dtype=np.uint8)
                image = image.reshape((height, width , channels))

                try:
                    self._buffer.put(image, timeout=buffer_put_timeout)
                except Full:
                    try: self._buffer.get_nowait()
                    except Empty: pass

                    try: self._buffer.put_nowait(image)
                    except Full: pass

        except Exception as ex:
            log.error(f"ffmpeg grab loop error: {ex}")

        finally:
            self._buffer = None
            process.terminate()
            process.kill()
            process.wait()
            log.info(f"ffmpeg disconnected from video devices: {self._device_paths}")

    def connect(self, buffer_size:int=1, frame_size:tuple[int, int]=(0,0), fps:float=0.0):

        if len(self._device_paths) == 0:
            raise ValidationError("Must define at least one input device")

        for device_path in self._device_paths.values():
            if not Path(device_path).exists(): 
                raise ValidationError(f"Video device '{device_path}' does not exist!")
        
        self._fps = int(fps) or 30
        self._frame_height = frame_size[0] or 480
        self._frame_width = frame_size[1] or 640
        
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

        self._thread_active = False
        self._fps = 0.0
        self._frame_width = 0
        self._frame_height = 0
        self._frame_count = 0
        self._total_frames = 0
    
    def source_name(self, side:int=None):
        return self._device_paths if side is None else self._device_paths[side]

    def get_frame(self) -> Frame:
        '''
        Returns a Frame with the image in RGB pixel format
        '''
        
        try:
            image = self._buffer.get() # waits indefinitely for a new image
        except:
            image = None

        frame = Frame(
            image, 
            len(self._device_paths), 
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

class VideoFile(FrameSource):

    def __init__(self, file_paths:dict[int, str]):

        self._file_paths = file_paths
        self._video_caps:dict[int, cv2.VideoCapture] = {}
        self._is_sbs = False

        self._fps = 0.0
        self._frame_width = 0
        self._frame_height = 0
        self._frame_count = 0
        self._total_frames = 0

    def connect(self, **kwargs):

        self._is_sbs = len(self._file_paths) == 2 and len(set(self._file_paths.values())) == 1
        if self._is_sbs:
            del self._file_paths[1]

        for file_path in self._file_paths.values():
            if not Path(file_path).exists(): 
                raise ValidationError(f"Video '{file_path}' does not exist!")
        
        for side, file_path in self._file_paths.items():
            self._video_caps[side] = cv2.VideoCapture(file_path)
            if not self._video_caps[side].isOpened():
                self.disconnect() # disconnect from all video caps
                raise ValidationError(f"Cannot open video: '{file_path}'")
        
        # validate that all sources have the same image shape and fps
        heights = []
        widths = []
        fpss =[]
        for video_cap in self._video_caps.values():
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

        for video_cap in self._video_caps.values():
            if video_cap:
                video_cap.release()
        self._video_caps = {}
        
        self._fps = 0.0
        self._frame_width = 0
        self._frame_height = 0
        self._frame_count = 0
        self._total_frames = 0
        log.info(f"disconnected from video devices: {self._file_paths}")

    def source_name(self, side:int=None):
        return self._file_paths if side is None else self._file_paths[side]

    def get_frame(self) -> Frame:
        '''
        Returns a Frame with the image in RGB pixel format
        '''

        images = []
        for side, video_cap in self._video_caps.items():
            success, image = video_cap.read()
            if success:
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)
            images.append(image)
            if not success or image is None:
                file_path = self._file_paths[side]
                log.info(f"Could not get image from file {file_path}. Reached end of video.")
            
        if len(images) == 1:
            image = images[0]  
        elif np.all(img is not None for img in images):
            image = np.hstack(images)
        else:
            image = None
        frame = Frame(
            image, 
            2 if self._is_sbs else len(images), 
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

def get_connected_uvc_cams():
    
    v4l_path = Path("/sys/class/video4linux")

    if not Path(v4l_path).exists(): 
        raise ValidationError(f"video4linux not in system")

    video_devices = {}
    for video_dir in v4l_path.iterdir():
        if not video_dir.is_dir():
            continue

        with open(video_dir / "index") as video_index_file:
            if int(video_index_file.read().strip()) != 0:
                continue
        
        with open(video_dir / "name") as video_name_file:
            device_path = f"/dev/{video_dir.name}"
            video_devices[device_path] = video_name_file.read().strip()

    return video_devices