from __future__ import annotations  # must be the first line of code

import numpy as np
import cv2
import supervision as sv
import base64
import numpy as np
from .utils import (
    sv_annotate, hex2rgb, COCO_CLASSES
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .trt import DetectionsBatch, Detections

class Frame:

    def __init__(self, image:np.ndarray|None, slices:int, frame_count:int, total_frames:int, source_fps:float):

        self.image:np.ndarray = image
        self._annotation_mask:np.ndarray = None

        self.slices = slices
        self.frame_count = frame_count
        self.total_frames = total_frames
        self.fps = source_fps

        self._label_annotator = sv.LabelAnnotator()
        self._box_annotator = sv.BoxAnnotator()
        self.detections_batch:DetectionsBatch = None

    @property
    def shape(self):
        return self.image.shape
    
    @property
    def height(self):
        return self.image.shape[0]

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def annotated_image(self):

        if self._annotation_mask is None:
            return self.image

        boolean_mask = np.any(self._annotation_mask != 0, axis=-1)
        image = self.image.copy()
        image[boolean_mask] = self._annotation_mask[boolean_mask]
        return image

    def split_image(self) -> list[np.ndarray]:
        """
        Splits the main image into several images according to the slices property and returns 
        a list of the slices
        """
        if self.image is None:
            return []
        split = self.width // self.slices
        if self.slices == 1:
            input_images = [self.image]
        else:
            input_images = []
            for i in range(self.slices):
                start = i * split
                end = start + split 
                input_images.append(self.image[:, start:end])

        return input_images

    def annotate(self, detections:Detections, slice:int, color_hex:str="#00FF00"):
                
        if self._annotation_mask is None:
            self._annotation_mask = np.zeros_like(self.image)
            
        split = self.width // self.slices
        split_start = split * slice
        split_end = split_start + split
        
        sv_annotate(
            self._annotation_mask[:, split_start:split_end], 
            detections.to_sv_detections(), 
            self._box_annotator, 
            color_rgb=hex2rgb(color_hex)
        )

        labels = [COCO_CLASSES[cls_id] for cls_id in detections.class_id]
        sv_annotate(
            self._annotation_mask[:, split_start:split_end], 
            detections.to_sv_detections(), 
            self._label_annotator, 
            color_rgb=hex2rgb(color_hex),
            labels=labels
        )
    
    def annotate_all(self):
        hex_colors = ["#00FF00", "#FF00FF", "#00FFFF"]
        for slice, dets in enumerate(self.detections_batch.detections):
            self.annotate(dets, slice, color_hex=hex_colors.pop(0))

    def resize(self, resize_factor:float):
        if resize_factor != 1.0:
            self.image = cv2.resize(
                self.image, None, dst=self.image, 
                fx=resize_factor, fy=resize_factor
            )

    def image_to_bytes(self) -> bytes:
        _, buffer = cv2.imencode('.jpg', self.image)
        return buffer.tobytes()

    def annotated_image_to_bytes(self) -> bytes:

        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(self.annotated_image, cv2.COLOR_RGB2BGR))
        return buffer.tobytes()

    def image_to_base64(self, ) -> str:
        _, buffer = cv2.imencode('.jpg', self.image)
        return base64.b64encode(buffer).decode('utf-8')

    def annotated_image_to_base64(self) -> str:

        _, buffer = cv2.imencode('.jpg', self.annotated_image)
        return base64.b64encode(buffer).decode('utf-8')
