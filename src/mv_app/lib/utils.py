
import cv2
import numpy as np
import supervision as sv
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from os.path import getsize as get_file_size
from enum import Enum
from urllib.parse import urlparse
import torch
import subprocess
import re
import os

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

def get_deepstream_version() -> str:
    
    version_file_path = Path("/opt/nvidia/deepstream/deepstream/version")
    
    if version_file_path.is_file():
        version = version_file_path.read_text().strip()
        match = re.match(r'Version:\s([\d\.]+)', version)
        if match:
            return match.group(1)
    return "-not found-"
        
def is_cuda_available():
    return torch.cuda.is_available()

def resource_usage_generator(interval_ms=200):
    """
    Generator for system metrics using tegrastats. 
    The generator yields a tuple as:
        cpu_percent, ram_percent, gpu_percent, vdd_value
    """
    process = subprocess.Popen(
        ["tegrastats", "--interval", str(interval_ms)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    os.set_blocking(process.stdout.fileno(), False) # pyright: ignore

    cpu_regex = re.compile(r"CPU\s\[(.+?)\]")
    cpu_cores_regex = re.compile(r"(\d+)%")
    ram_regex = re.compile(r"RAM\s(\d+)/(\d+)MB")
    gpu_regex = re.compile(r"GR3D_FREQ\s(\d+)%")
    vdd_regex = re.compile(r"VDD_IN\s(\d+)mW/(\d+)mW/(\d+)mW") # groups 1,2,3= current, average, peak 
    
    def parse_tegrastats(line:str):

        cpu_match = cpu_regex.search(line)
        gpu_match = gpu_regex.search(line)
        ram_match = ram_regex.search(line)
        vdd_match = vdd_regex.search(line)
        
        if not (ram_match and cpu_match and gpu_match and vdd_match):
            return None

        # 10-29-2026 08:16:11 RAM 2374/7546MB (lfb 7x4MB) SWAP 0/16384MB (cached 0MB) CPU [0%@729,0%@729,0%@729,0%@729,0%@729,0%@729] GR3D_FREQ 0% cpu@55.312C/55.625C soc2@55.468C/55.562C soc0@54.906C/55.093C gpu@55.562C/55.906C tj@55.906C/56.25C soc1@55.906C/56.25C VDD_IN 5475mW/5432mW/5475mW VDD_CPU_GPU_CV 1426mW/1415mW/1426mW VDD_SOC 1349mW/1349mW/1349mW
        ram_used = int(ram_match.group(1))
        ram_total = int(ram_match.group(2))
        ram_percent = (ram_used / ram_total) * 100 if ram_total > 0 else 0.0
    
        cpu_values = [int(x) for x in cpu_cores_regex.findall(cpu_match.group(1))]
        cpu_percent = float(np.mean(cpu_values)) if cpu_values else 0.0
    
        gpu_percent = int(gpu_match.group(1))

        vdd_value = int(vdd_match.group(1))
    
        return cpu_percent, ram_percent, gpu_percent, vdd_value

    def get_last_line():

        last_line = ""
        while True:
            line:str = process.stdout.readline()
            if not line:
                break
            last_line = line.strip()
        return last_line

    try:
        while True:
            line = get_last_line()
            if not line:
                continue
            values = parse_tegrastats(line)
            cpu_percent, ram_percent, gpu_percent, vdd_value = values if values else (-1, -1, -1, -1)
            yield cpu_percent, ram_percent, gpu_percent, vdd_value

    finally:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
