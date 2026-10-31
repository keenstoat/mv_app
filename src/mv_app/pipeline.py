
# from ultralytics.models.yolo.model import YOLO
import traceback
from pathlib import Path
import logging
log = logging.getLogger(__name__)
import torch
import gc
import time
from urllib.parse import urlparse

from trackers import ByteTrackTracker


from .lib.trt import TensorRT
from .lib.frame import Frame
from .lib.fps_monitor import FPSMonitor
from .lib.frame_sources import (UVCCamCv2, UVCCamGst, VideoFile, FrameSource)
from .lib.stereo_depth import StereoDepth
from .lib.utils import (
    ValidationError,
    MsgType, RunState,
    is_cuda_available, get_tensorrt_version,
)

class Pipeline:
    
    def __init__(self):

        self.conf_frame_source_urls:list[str] = []
        self.frame_source:FrameSource = None # pyright: ignore
                
        self.conf_grab_tool = 'gst'
        self.conf_grab_buffer_size = 1
        self.conf_resize_factor = 1.0
        self.conf_cam_frame_width = 640
        self.conf_cam_frame_height = 480
        self.conf_cam_fps = 30

        self.conf_inference_method = 'none'
        self.vision_model:TensorRT = None # pyright: ignore

        self.conf_depth_analysis_method = 'none'

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
        self.vision_model = None # pyright: ignore
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

            raise ValidationError(f"Grap tool '{self.conf_grab_tool}' not valid", MsgType.ERROR)
        raise ValidationError(f"Source '{url_schemes}' not valid", MsgType.ERROR)
    
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
                return "Must stop pipeline before starting a new run", MsgType.WARNING

            if not self.conf_frame_source_urls:
                return "Must define at least one frame source", MsgType.WARNING
                
            if self.conf_inference_method != 'none' and self.vision_model is None:
                return "Must load a vision model for inference", MsgType.WARNING

            if self.conf_depth_analysis_method != 'none':

                stereo_rectify_data_file = Path(__file__).parent / "lib" / "stereo_data"
                filename = f"stereo_rectify_{self.conf_cam_frame_width}x{self.conf_cam_frame_height}.pickle"
                stereo_rectify_data_file = stereo_rectify_data_file / filename
                if not stereo_rectify_data_file.is_file():
                    raise ValidationError(f"There is no file for the given frame dimensions: {filename}")

                try:
                    stereo_rectifier = StereoDepth(stereo_rectify_data_file)
                except:
                    raise ValidationError(f"Stereo rectification maps failed to load from file {stereo_rectify_data_file}")


            if self.conf_depth_analysis_method == 'triangulate' and self.conf_inference_method == 'none':
                return "Triangulate depth analysis requires inference enabled", MsgType.WARNING
            
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
                            if self.conf_depth_analysis_method != 'none':
                                stereo_rectifier.rectify_gpu(frame)
                            return frame

                        predict_generator = self.vision_model.predict_optimized(
                            get_next_frame=get_next_frame,
                            conf=0.5
                        )
                    
                    frame = next(predict_generator)

                elif self.conf_inference_method == 'simple':

                    frame = self.frame_source.get_frame()

                    if self.conf_depth_analysis_method != 'none':
                        stereo_rectifier.rectify_gpu(frame)

                    self.vision_model.predict(frame, conf=0.5)
                
                else:
                    frame = self.frame_source.get_frame()
                    if not frame.is_valid:
                        return "No more images in source. End of video stream.", MsgType.WARNING

                # DEPTH ESTIMATION -----------------------------------------------------------

                if self.conf_depth_analysis_method == 'triangulate':

                    # this is a double check only
                    if not frame.is_rectified or frame.detections_batch is None:
                        raise RuntimeError("Triangulation requires rectification and detection")
    
                    for detections in frame.detections_batch.detections_list:
                        # update the detections with their tracker id
                        # tracker will treat images as a sequence of teach other
                        sv_tracked_detections = tracker.update(detections.to_sv_detections())
                        detections.update_with_sv_detections(sv_tracked_detections)

                    frame.detections_batch.order_detctions_by_tracker_id()
                    
                    stereo_rectifier.triangulate(frame)
                    frame.annotate_all()

                    # use this to show only one frame
                    # frame.image_tensors = [frame.image_tensors[0]]
                    # frame.annotate(frame.detections_batch.detections_list[0], 0)

                    
                elif self.conf_depth_analysis_method == 'disparity':

                    # this is a double check only
                    if not frame.is_rectified:
                        stereo_rectifier.rectify_gpu(frame)

                    stereo_rectifier.dense_disparity(frame)
                    frame.image_tensors = [frame.dense_disparity_depth]

                else:
                    frame.annotate_all()
                    
                self.display_frame = frame

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

            self.frame_source = None # pyright: ignore
            self.display_frame = None # pyright: ignore
            self.conf_frame_source_urls = []
            self.clean_cuda()
            self.stop()
