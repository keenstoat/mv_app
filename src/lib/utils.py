
import cv2
import numpy as np
import supervision as sv
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from os.path import getsize as get_file_size
from enum import Enum
from urllib.parse import urlparse
import psutil
import torch
import platform
import subprocess
import re

COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

_base_dir = Path(__file__).parent.absolute()

# constants and enums =========================================================================================


class MsgType(Enum):
    SUCCESS = "positive"
    WARNING = "warning"
    ERROR = "negative"

class RunState(Enum):
    IDLE = "IDLE"
    PROCESSING = "PROCESSING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"

# classes ======================================================================================================

class ValidationError(Exception):
    """
    This class used to raise expected exceptions. For example, when validating inputs in a function, instead of an 'assert' raise a ValidationError. This allows you to distinguish your validation from a common exception  in your try-except block.
    """
    def __init__(self, message:str, message_type=MsgType.ERROR):
        self.message = message
        self.message_type = message_type
    
    def __str__(self):
        return self.message

# annotations functions ===========================================================================================

def hex2rgb(hex_color:str):
    """
    Convert #RRGGBB color in hex format to an RGB tuple 
    """
    hex_color = hex_color.strip().lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def hex2bgr(hex_color:str):
    """
    Convert #RRGGBB color in hex format to an BGR tuple 
    """
    return hex2rgb(hex_color)[::-1]

def draw_text(image:np.ndarray, text:str, text_pos:tuple[int, int], size=10, 
    color_rgb:tuple[int, int, int]=(255, 0, 255), bg_color_rgb:tuple[int, int, int]=None):
    
    """
    Draws the given text on the image, using the UbuntuMono font.
    """
    font_file = _base_dir / "fonts" / "UbuntuMono.ttf"

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_file, size=size)

    x, y = text_pos
    if bg_color_rgb:
        bg_rect = draw.textbbox((x, y), text, font=font)
        draw.rectangle(bg_rect, fill=bg_color_rgb)
    
    draw.text((x, y), text, font=font, fill=color_rgb)

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def sv_annotate(image:np.ndarray, detections:sv.Detections, 
    annotator, color_rgb:tuple[int, int, int]=(0, 255, 0), thickness=0, labels:list[str]=[]):
    """
    Annotates the supervision Detections on the image using the given annotator and color.
    If not provided, the line thickness is automatically determined from the image size.
    """
    if len(detections.xyxy) > 0:

        if hasattr(annotator, "thickness"):
            annotator.thickness = thickness or max(1, int(max(image.shape[0:2]) * 0.003))
        
        if hasattr(annotator, "color"):
            annotator.color = sv.Color(*color_rgb)

        if hasattr(annotator, "text_color"):
            annotator.text_color = sv.Color(0,0,0)

        if labels:
            annotator.annotate(image, detections=detections, labels=labels)
        else:
            annotator.annotate(image, detections=detections)

    return image

# pre-test validation functions =================================================================================

def get_mean_brightness(image:np.ndarray):
    """
    Returns the mean brightness of the image. The value is normalized between 0 and 1.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(image) / 255, np.std(image) / 255

# other functions ===============================================================================================

def get_icon_path(icon_name):
    return str(_base_dir / "icons" / f"{icon_name}.svg")

def is_url(url) -> bool:
    url = urlparse(url)
    return bool(url.scheme and url.hostname)

def sort_filepaths_by_file_size(filepath_list:list[str]):
    filepath_list = [(f, get_file_size(f)) for f in filepath_list]
    filepath_list = sorted(filepath_list, key=lambda x: x[1])
    return [filepath for filepath, _ in filepath_list]

def get_tensorrt_version() -> str:
    import tensorrt
    return tensorrt.__version__

def get_cuda_version() -> str:
    return torch.version.cuda

def is_cuda_available():
    return torch.cuda.is_available()

def get_cpu_percent() -> float:
    return psutil.cpu_percent(interval=None)

def get_ram_percent() -> float:
    return psutil.virtual_memory().percent

def get_gpu_percent() -> float:

    if "tegra" not in platform.release():
        return -1.0

    try:
        process = subprocess.Popen(['tegrastats', '--interval', '3'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        first_line = process.stdout.readline()
        process.terminate()
        process.kill()

        gpu_match = re.search(r'GR3D_FREQ\s+(\d+)%', first_line)
        if gpu_match:
            return float(gpu_match.group(1))
        else:
            return -2.0
    except Exception as e:
        return -3.0
