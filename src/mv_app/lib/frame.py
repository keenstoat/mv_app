from __future__ import annotations  # must be the first line of code

import numpy as np
import cv2
import torch
import supervision as sv
import base64
from pathlib import Path
import time
from .utils import (
    sv_annotate, hex2rgb, COCO_CLASSES
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .trt import DetectionsBatch, Detections

class Frame:

    def __init__(self, images:list[np.ndarray], frame_count:int, total_frames:int, source_fps:float):

        self.images:list[np.ndarray] = images
        self._image_tensors:list[torch.Tensor] = []
        self._annotation_masks:list[np.ndarray] = []
        self.is_rgb = True
        self.is_rectified = False

        self.frame_count = frame_count
        self.total_frames = total_frames
        self.fps = source_fps

        self._label_annotator = sv.LabelAnnotator()
        self._box_annotator = sv.BoxAnnotator()
        self.detections_batch:DetectionsBatch|None = None

    @property
    def is_valid(self):
        return self.slices > 0 and all(img is not None for img in self.images)
        
    @property
    def slices(self):
        return len(self.images)
    
    @property
    def shape(self):
        return self.images[0].shape
    
    @property
    def height(self):
        return self.images[0].shape[0]

    @property
    def width(self):
        return self.images[0].shape[1]

    @property
    def tensors(self) -> list[torch.Tensor]:

        if self._image_tensors:
            return self._image_tensors
            
        for image in self.images:
            image_tensor = torch.as_tensor(image.copy(), dtype=torch.uint8, device='cuda')
            self._image_tensors.append(image_tensor)
        return self._image_tensors

    @property
    def annotation_masks(self) -> list[np.ndarray]:

        if not self._annotation_masks:
            self._annotation_masks = [np.zeros_like(image) for image in self.images]

        return self._annotation_masks

    @property
    def annotated_image(self) -> np.ndarray:
        """
        Returns the image with drawn annotations. Does not modify the original images.
        """
        if self.is_rectified:

            image = torch.hstack(self._image_tensors).cpu().numpy()
        else:
            image = np.hstack(self.images)

        # use the underlaying attribute, if annotation masks is empty, then 
        # skip the operation and just return the image
        if not self._annotation_masks:
            return image

        # np.hstack creates a copy of the arrays, so its ok to modify with the mask
        annotation_mask = np.hstack(self._annotation_masks)
        boolean_mask = np.any(annotation_mask != 0, axis=-1)
        image[boolean_mask] = annotation_mask[boolean_mask]

        return image

    def annotate(self, detections:Detections, image_idx:int, color_hex:str="#00FF00"):

        sv_detections = detections.to_sv_detections()
        sv_annotate(
            self.annotation_masks[image_idx], 
            sv_detections, 
            self._box_annotator, 
            color_rgb=hex2rgb(color_hex)
        )
        if detections.tracker_id is None:
            labels = [COCO_CLASSES[class_id] for class_id in detections.class_id]
        else:
            labels = [
                f"{COCO_CLASSES[class_id]} - {tracker_id} - {depth_m:.2f}"
                for tracker_id, class_id, depth_m
                in zip(detections.tracker_id, detections.class_id, detections.metadata)
            ]
        sv_annotate(
            self.annotation_masks[image_idx], 
            sv_detections, 
            self._label_annotator, 
            color_rgb=hex2rgb(color_hex),
            labels=labels
        )
    
    def annotate_all(self):
        
        hex_colors = ["#00FF00", "#FF00FF", "#00FFFF"]

        if self.detections_batch is None:
            return
            
        for image_idx, detections_batch in enumerate(self.detections_batch.detections_list):
            self.annotate(detections_batch, image_idx, color_hex=hex_colors.pop(0))

    def image_to_bytes(self) -> bytes:
        image = cv2.cvtColor(np.hstack(self.images), cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', image)
        return buffer.tobytes()

    def image_to_base64(self, ) -> str:
        image = cv2.cvtColor(np.hstack(self.images), cv2.COLOR_RGB2BGR)
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

        image = np.hstack(self.images)
        image_path = save_dir / filename
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
