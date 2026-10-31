from __future__ import annotations  # must be the first line of code

import numpy as np
import cv2
import torch
import supervision as sv
import base64
from pathlib import Path
import time
from contextlib import contextmanager
from .utils import (
    sv_annotate, hex2rgb, COCO_CLASSES
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .trt import DetectionsBatch, Detections

class Frame:

    def __init__(self, image_tensors:list[torch.Tensor], 
        frame_count:int=-1, total_frames:int=-1, source_fps:float=-1.0):

        self.image_tensors:list[torch.Tensor] = image_tensors
        self._image_arrays:list[np.ndarray] = []
        self._annotation_masks:list[np.ndarray] = []
        self.is_rgb = True
        self.layout = "HWC" # same as cv2 np array images
        self.is_rectified = False

        self.frame_count = frame_count
        self.total_frames = total_frames
        self.fps = source_fps

        self._label_annotator = sv.LabelAnnotator()
        self._box_annotator = sv.BoxAnnotator()
        self.detections_batch:DetectionsBatch|None = None

        self.dense_disparity_depth:torch.Tensor = None 

        self.perf:dict[str, float] = {}

    @property
    def is_valid(self):
        # is a valid frame if there is at least one image and none of the images are None
        return self.slices > 0 and all(img is not None for img in self.image_tensors)
        
    @property
    def slices(self):
        return len(self.image_tensors)
    
    @property
    def shape(self):
        return tuple(self.image_tensors[0].shape)
    
    @property
    def height(self):
        return self.image_tensors[0].shape[0] # torch tensor is HWC

    @property
    def width(self):
        return self.image_tensors[0].shape[1] # torch tensor is HWC

    @property
    def image_arrays(self) -> list[np.ndarray]:

        if self._image_arrays:
            return self._image_arrays
            
        for tensor in self.image_tensors:
            self._image_arrays.append(tensor.cpu().numpy())
        return self._image_arrays

    @property
    def annotation_masks(self) -> list[np.ndarray]:

        if not self._annotation_masks:
            self._annotation_masks = [np.zeros_like(image) for image in self.image_arrays]

        return self._annotation_masks

    @property
    def annotated_image(self) -> np.ndarray:
        """
        Returns the image with drawn annotations. Does not modify the original images.
        """

        image = np.hstack(self.image_arrays)

        # use the underlaying attribute, if annotation masks is empty, then 
        # skip the operation and just return the image
        if not self._annotation_masks:
            return image

        # np.hstack creates a copy of the arrays, so its ok to modify with the mask
        annotation_mask = np.hstack(self.annotation_masks)
        boolean_mask = np.any(annotation_mask != 0, axis=-1)
        image[boolean_mask] = annotation_mask[boolean_mask]

        return image

    def annotate(self, detections:Detections, slice:int, color_hex:str="#00FF00"):

        sv_detections = detections.to_sv_detections()

        # annotate bounding boxes
        sv_annotate(
            self.annotation_masks[slice], 
            sv_detections, 
            self._box_annotator, 
            color_rgb=hex2rgb(color_hex)
        )

        # annotate labels
        if detections.tracker_id is None:
            labels = [COCO_CLASSES[class_id] for class_id in detections.class_id]
        elif detections.xyz is None:
            labels = [
                f"{COCO_CLASSES[class_id]} - {tracker_id}"
                for tracker_id, class_id
                in zip(detections.tracker_id, detections.class_id)
            ]
        else:
            labels = [
                f"{COCO_CLASSES[class_id]} - {tracker_id} - {xyz[2]:.2f} mm"
                for tracker_id, class_id, xyz
                in zip(detections.tracker_id, detections.class_id, detections.xyz)
            ]
        sv_annotate(
            self.annotation_masks[slice], 
            sv_detections, 
            self._label_annotator, 
            color_rgb=hex2rgb(color_hex),
            labels=labels
        )

        # annotate depth
    
    def annotate_all(self):
        
        hex_colors = ["#00FF00", "#FF00FF", "#00FFFF"]

        if self.detections_batch is None:
            return
            
        for slice, detections_batch in enumerate(self.detections_batch.detections_list):
            self.annotate(detections_batch, slice, color_hex=hex_colors.pop(0))

    def image_to_bytes(self) -> bytes:
        
        image = cv2.cvtColor(np.hstack(self.image_arrays), cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', image)
        return buffer.tobytes()

    def image_to_base64(self, ) -> str:
        image = cv2.cvtColor(np.hstack(self.image_arrays), cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
        
    def annotated_image_to_bytes(self) -> bytes:
        image = cv2.cvtColor(self.annotated_image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', image)
        return buffer.tobytes()

    def annotated_image_to_base64(self) -> str:
        image = cv2.cvtColor(self.annotated_image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def save_image(self, save_dir_path:str, filename:str=""):
        
        save_dir = Path(save_dir_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        if not filename:
            filename = f"{int(time.time())}_{self.width}x{self.height}.jpg"

        image = np.hstack(self.image_arrays)
        image_path = save_dir / filename
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    @contextmanager
    def time_it(self, key:str):
        """
        Records the time taken for the executed block (yield). Adds the time if the key already exists
        """
        t0 = time.perf_counter()
        yield 
        tf = (time.perf_counter() - t0) * 1000
        self.perf[key] = self.perf[key] + tf if key in self.perf else tf
        
