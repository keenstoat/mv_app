
# from ultralytics.models.yolo.model import YOLO
import traceback
from pathlib import Path
import logging
log = logging.getLogger(__name__)
import torch
import numpy as np
import gc
import time
from urllib.parse import urlparse

from trackers import ByteTrackTracker


from lib.trt import TensorRT, Detections
from lib.frame import Frame
from lib.fps_monitor import FPSMonitor
from lib.frame_sources import (UVCCamCv2, UVCCamGst, VideoFile, FrameSource)
from lib.rectifier import StereoRectifier
from lib.utils import (
    ValidationError,
    MsgType, RunState,
    is_cuda_available, get_tensorrt_version,
)

class Pipeline:
    
    def __init__(self):

        self.conf_frame_source_urls:list[str] = []
        self.frame_source:FrameSource|None = None
                
        self.conf_grab_tool = 'gst'
        self.conf_grab_buffer_size = 1
        self.conf_resize_factor = 1.0
        self.conf_cam_frame_width = 640
        self.conf_cam_frame_height = 480
        self.conf_cam_fps = 30

        self.conf_inference_method = 'none'
        self.vision_model:TensorRT = None # pyright: ignore

        self.fps_monitor:FPSMonitor = None # pyright: ignore
        self.display_frame:Frame = None # pyright: ignore

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
        self.clean_cuda()
        self.vision_model = None
        log.info("model unloaded")

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
                msg = "TensorRT engine can only be used with TensorRT and CUDA available"
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
        url_schemes = {urlparse(url).scheme for url in self.conf_frame_source_urls}
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

            if self.conf_inference_method != 'none' and self.vision_model is None:
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

            calibration_data_path = "/home/charles/repos/mv_app/tools/cam_calibration/calibration_data.pickle"
            stereo_rectifier = StereoRectifier(calibration_data_path)

            # optimized inferece uses these
            predict_generator = None
            
            tracker = ByteTrackTracker(
                lost_track_buffer=10,
                track_activation_threshold=0.5,
                minimum_iou_threshold=0.1,
                
            )

            while self.is_processing() or self.is_paused():

                if self.is_paused():
                    time.sleep(0.1)
                    continue

                self.fps_monitor.tick()

                # INFERENCE ------------------------------------------------------------------
                if self.conf_inference_method == 'optimized':

                    if predict_generator is None:
                        
                        def get_next_frame():
                            frame = self.frame_source.get_frame()
                            stereo_rectifier.rectify(frame)
                            return frame

                        predict_generator = self.vision_model.predict_optimized(
                            get_next_frame=get_next_frame,
                            conf=0.5
                        )
                    
                    frame = next(predict_generator)

                elif self.conf_inference_method == 'simple':

                    frame = self.frame_source.get_frame()
                    stereo_rectifier.rectify(frame)
                    self.vision_model.predict(frame, conf=0.5)
                
                else:
                    frame = self.frame_source.get_frame()
                    if not frame.is_valid:
                        return "No more images in source. EOV", MsgType.WARNING

                # DEPTH ESTIMATION -----------------------------------------------------------
                if False and frame.detections_batch:

                    track_history:dict[int, list[Detections]] = {}
                    for detections in frame.detections_batch.detections_list:
                        # update the detections with their tracker id
                        # tracker will treat images as a sequence of teach other
                        tracked_detections = tracker.update(detections.to_sv_detections())
                        detections.update_with_sv_detections(tracked_detections)

                        # pair detections by their tracker id
                        if detections.tracker_id.size == 0 :
                            continue
                        
                        for track_id, xyxy, class_id in zip(detections.tracker_id, detections.xyxy, detections.class_id): 
                            if track_id < 0:
                                continue
                            if track_id not in track_history:
                                track_history[track_id] = []
                                
                            track_history[track_id].append(
                                Detections(
                                    xyxy=np.array([xyxy]), 
                                    tracker_id=np.array([track_id]),
                                    class_id = np.array([class_id])
                                )
                            )
                            
                    frame.images = frame.split_image()[0]
                    frame.slices = 1
                    focal_px = 600 # px
                    baseline_m = 0.05 # 5 cm
                    for detection_pair in track_history.values():
                        if len(detection_pair) < 2:
                            continue
                        lx1, _, lx2, _ = detection_pair[0].xyxy[0]
                        rx1, _, rx2, _ = detection_pair[1].xyxy[0]
                    
                        # Bounding box centers
                        x_left = (lx1 + lx2) / 2
                        x_right = (rx1 + rx2) / 2
                    
                        disparity = x_left - x_right

                        depth_m = 100
                        if disparity > 0:
                            depth_m = focal_px * baseline_m / disparity
                        detection_pair[0].metadata.append(depth_m)
                        frame.annotate(detection_pair[0], 0)

                frame.annotate_all()
                self.display_frame = frame

                # STEP 5: Execute the call back
                if callback_func is not None:
                    callback_func()
    
        except ValidationError as ve:
            log.warning(ve.message)
            return ve.message, ve.message_type

        except Exception as ex:
            log.error(traceback.format_exc())
            return str(ex), MsgType.ERROR

        finally:
            if self.frame_source:
                self.frame_source.disconnect()

            self.frame_source = None
            self.display_frame = None
            self.conf_frame_source_urls = []
            self.clean_cuda()
            self.stop()
