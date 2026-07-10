
from ultralytics.models.yolo.model import YOLO
import traceback
import cv2
from pathlib import Path
import logging as log
import torch
import gc
import time
from urllib.parse import urlparse
import numpy as np

from lib.trt import TensorRT
from lib.frame import Frame
from lib.fps_monitor import FPSMonitor
from lib.frame_sources import (UVCCamCv2, UVCCamGst, UVCCamFFmpeg, VideoFile, FrameSource)
from lib.utils import (
    ValidationError,
    MsgType, RunState,
    is_cuda_available, get_tensorrt_version,
)

class Pipeline:
    
    def __init__(self):

        self.conf_frame_source_urls:list[str] = []
        self.frame_source:FrameSource = None
                
        self.conf_grab_tool = 'gst'
        self.conf_grab_buffer_size = 1
        self.conf_resize_factor = 1.0
        self.conf_cam_frame_width = 640
        self.conf_cam_frame_height = 480
        self.conf_cam_fps = 30

        self.conf_enable_inference = False
        self.conf_use_cvcuda = True
        self.conf_use_optimized_inference = True
        self.vision_model:TensorRT = None 

        self.fps_monitor:FPSMonitor = None
        self.display_frame:Frame = None

        self.run_state = RunState.STOPPED
    
    def stop(self):
        self.run_state = RunState.STOPPED

    def pause(self):
        self.run_state = RunState.PAUSED

    def resume(self):
        self.run_state = RunState.PROCESSING

    def unload_vision_model(self):
        """
        Unloads the yolo model from memory. After calling this method, the model must be loaded again.
        """
        if self.vision_model is None:
            return

        del self.vision_model
        if is_cuda_available(): 
            torch.cuda.empty_cache()
        gc.collect()
        self.vision_model = None
        log.info(f"model unloaded" )

    def load_vision_model(self, model_filepath:Path):
        """
        Loads the model at the given filepath. The model must be loaded before running any tests with it.
        The test methods will halt if the model is not loaded previously. 
        This is done to avoid loading the model when the test starts running, as it creates a tiny delay in execution while the model loads. There is no error, it just looks bad.
        """

        if self.vision_model is not None:
            msg = "Must unload the current model before loading a new one"
            raise ValidationError(msg, MsgType.WARNING)

        if not model_filepath.exists():
            msg = f"Model file '{model_filepath}' does not exist."
            raise ValidationError(msg, MsgType.WARNING)

        if model_filepath.suffix == ".engine":
            if not is_cuda_available() and not get_tensorrt_version():
                msg = f"TensorRT engine can only be used with TensorRT and CUDA available"
                raise ValidationError(msg, MsgType.WARNING)
            self.vision_model = TensorRT(model_filepath)
            log.info(f"loaded model '{model_filepath}'" )
            
        # elif model_filepath.suffix == ".pt":
        #     self.vision_model = YOLO(model_filepath, task='detect')
        #     if is_cuda_available():
        #         self.vision_model.to('cuda')
        #     log.info(f"loaded model '{model_filepath}'" )

        # elif model_filepath.suffix == ".onnx":
        #     # onnx files were found to run faster than plain pt, but it needs 'onnxruntime-gpu'. The device used
        #     # will be auto selected by YOLO, which fails to explicitly do .to('cuda') on onnx models.
        #     self.vision_model = YOLO(model_filepath, task='detect')
        #     log.info(f"loaded model '{model_filepath}'" )

        else:
            log.error(f"Model '{model_filepath}' not supported. Acceptable model formats are .engine")
            raise ValidationError(f"Model '{model_filepath}' not supported")

    def select_frame_source(self) -> FrameSource:
        
        if not self.conf_frame_source_urls:
            raise ValidationError("Must define at least one video source string", MsgType.WARNING)

        # resolve the urls schemes to a set, so duplicates are removed
        url_schemes = set(urlparse(url).scheme for url in self.conf_frame_source_urls)
        if len(url_schemes) != 1:
            raise ValidationError("All frame source must be of the same type", MsgType.WARNING)

        url_schemes = url_schemes.pop()
        if url_schemes == "file":
            file_paths = [path.removeprefix("file://") for path in self.conf_frame_source_urls]
            return VideoFile(file_paths)

        if url_schemes == "uvc":
            device_paths = [path.removeprefix("uvc://") for path in self.conf_frame_source_urls]
            if self.conf_grab_tool == 'gst':
                return UVCCamGst(device_paths)
            
            if self.conf_grab_tool == 'cv2':
                return UVCCamCv2(device_paths)
            
            if self.conf_grab_tool == 'ffmpeg':
                return UVCCamFFmpeg(device_paths)

        raise ValidationError(f"Source '{url_schemes}' not valid", MsgType.WARNING)
    
    def is_processing(self):
        return self.run_state == RunState.PROCESSING

    def is_stopped(self):
        return self.run_state == RunState.STOPPED

    def is_paused(self):
        return self.run_state == RunState.PAUSED
    
    def clean_cuda(self):
        if is_cuda_available(): 
            torch.cuda.empty_cache()
        gc.collect()
        
    def process(self, callback_func=None) -> tuple[str, MsgType] | None:
        
        try:
            
            if not self.is_stopped():
                return "Must stop pipeline", MsgType.WARNING

            if self.conf_enable_inference and self.vision_model is None:
                return "Must load a vision model for inference", MsgType.WARNING

            if not self.conf_frame_source_urls:
                return "Must define at least one frame source", MsgType.WARNING

            
            self.frame_source = self.select_frame_source()
            self.frame_source.connect(
                buffer_size=self.conf_grab_buffer_size, 
                frame_size=(self.conf_cam_frame_height, self.conf_cam_frame_width), 
                fps=self.conf_cam_fps
            )

            self.fps_monitor = FPSMonitor()
            self.fps_monitor.set_target_fps(self.frame_source.fps)

            self.run_state = RunState.PROCESSING

            # optimized inferece uses these
            predict_generator = None
            frame_queue:list[Frame] = []

            while self.is_processing() or self.is_paused():

                if self.is_paused():
                    time.sleep(0.1)
                    continue

                self.fps_monitor.tick()

                # INFERENCE ------------------------------------------------------------------
                if self.conf_enable_inference:

                    if self.conf_use_optimized_inference:

                        if predict_generator is None:
                            
                            def get_frame_split_image():
                                frame_queue.append(self.frame_source.get_frame())
                                return frame_queue[-1].split_image()

                            predict_generator = self.vision_model.predict_generator(
                                get_images=get_frame_split_image,
                                conf=0.5, 
                                use_cvcuda=self.conf_use_cvcuda
                            )
                        
                        detections_batch, _ = next(predict_generator)
                        frame = frame_queue.pop(0)
                        frame.detections_batch = detections_batch

                    else:
                        frame = self.frame_source.get_frame()
                        detections_batch = self.vision_model.predict(
                            frame.split_image(),
                            conf=0.5, 
                            use_cvcuda=self.conf_use_cvcuda
                        )
                        frame.detections_batch = detections_batch
                    
                    frame.annotate_all()
                
                else:
                    frame = self.frame_source.get_frame()
                    if frame.image is None:
                        return "No more images in source. EOV", MsgType.WARNING

                self.display_frame = frame

                # STEP 5: Execute the call back
                if callback_func is not None:
                    callback_func()
    
        except ValidationError as ve:
            log.warning(ve.message)
            return ve.message, ve.message_type

        except Exception as ex:
            log.error(traceback.format_exc())
            return ex, MsgType.ERROR

        finally:
            if self.frame_source:
                self.frame_source.disconnect()

            self.frame_source = None
            self.display_frame = None
            self.conf_frame_source_urls = []
            self.clean_cuda()
            self.stop()

