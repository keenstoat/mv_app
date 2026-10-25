import numpy as np
import cv2
import torch
import cvcuda
import pickle
from pathlib import Path
import time
from .frame import Frame

class StereoRectifier:

    def __init__(self, calibration_data_filepath:str) -> None:

        calibration_data_path = Path(calibration_data_filepath)
        calibration_data_dict = pickle.loads(calibration_data_path.read_bytes())
    
        self.map_0_x = calibration_data_dict["map_0_x"]
        self.map_0_y = calibration_data_dict["map_0_y"]
        self.map_1_x = calibration_data_dict["map_1_x"]
        self.map_1_y = calibration_data_dict["map_1_y"]

        # used for GPU rectification
        map_0 = np.stack((self.map_0_x, self.map_0_y), axis=-1).astype(np.float32)
        self.map_0 = torch.as_tensor(map_0, dtype=torch.float32, device='cuda')

        map_1 = np.stack((self.map_1_x, self.map_1_y), axis=-1).astype(np.float32)
        self.map_1 = torch.as_tensor(map_1, dtype=torch.float32, device='cuda')

    def rectify(self, frame:Frame):
        """
        Calculates the stereo rectification of the images and replaces the images.
        The rectification is done over the np arrays on cpu.
        """
        
        map_list = [(self.map_0_x, self.map_0_y), (self.map_1_x, self.map_1_y)]
        for idx, map in enumerate(map_list):
            frame.images[idx] = cv2.remap(frame.images[idx], *map, cv2.INTER_LINEAR)

    def rectify_gpu(self, frame:Frame):
        """
        Calculates the stereo rectification of the image tensors.
        The rectification is done over the torch tensors on gpu.
        The original images (np arrays) remain unchanged.
        The image in the rectified tensors differ from the original images
        """
        
        map_list = [self.map_0, self.map_1]
        for idx, map in enumerate(map_list):
            image_tensor = cvcuda.remap(
                src=cvcuda.as_tensor(frame.tensors[idx], "HWC"),
                map=cvcuda.as_tensor(map, "HWC"),
                src_interp=cvcuda.Interp.LINEAR,
                map_interp=cvcuda.Interp.NEAREST,
                map_type=cvcuda.Remap.ABSOLUTE,
                border=cvcuda.Border.CONSTANT,
            )
            image_tensor = torch.as_tensor(image_tensor.cuda(), dtype=torch.uint8, device='cuda')
            frame.tensors[idx] = image_tensor
            frame.is_rectified = True



if __name__ == "__main__":

    _calibration_data_path = "/home/charles/repos/mv_app/tools/cam_calibration/calibration_data.pickle"

    _stereo_rectifier = StereoRectifier(_calibration_data_path)

    _image_path = "/home/charles/repos/mv_app/tools/cam_calibration/images/1792802281_640x480.jpg"
    _image = cv2.imread(_image_path)

    # CPU benchmark
    for _ in range(50):
        _stereo_rectifier.rectify_cpu(_image, _image)
        
    t0 = time.perf_counter()
    for _ in range(100):
        _rect_image, _ = _stereo_rectifier.rectify_cpu(_image, _image)
    print("CPU: ", (time.perf_counter() - t0) * 1000, "ms")
    cv2.imwrite("rectified-cpu.jpg", _rect_image)
    

    # GPU benchmark
    _image_tensor = torch.as_tensor(_image, dtype=torch.float32, device='cuda')
    with cvcuda.Stream():
        for _ in range(50):
            
            _stereo_rectifier.rectify(_image_tensor, _image_tensor)
        cvcuda.Stream().sync()
    
        t0 = time.perf_counter()
        for _ in range(100):
            _tensor_image_0, _ = _stereo_rectifier.rectify(_image_tensor, _image_tensor)
            cvcuda.Stream().sync()
    print("GPU: ", (time.perf_counter() - t0) * 1000, "ms")
    _rect_image = _tensor_image_0.cpu().numpy()
    cv2.imwrite("rectified-gpu.jpg", _rect_image)

