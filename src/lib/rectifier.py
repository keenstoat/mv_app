import numpy as np
import cv2
import torch
import cvcuda

class StereoRectifier:

    def __init__(self, map_0_x, map_0_y, map_1_x, map_1_y) -> None:

        # used for CPU rectification
        self.map_0_x = map_0_x 
        self.map_0_y = map_0_y 
        self.map_1_x = map_1_x 
        self.map_1_y = map_1_y

        # used for GPU rectification
        map_0 = np.stack((map_0_x, map_0_y), axis=-1).astype(np.float32)
        self.map_0 = torch.as_tensor(map_0, dtype=torch.float32, device='cuda')

        map_1 = np.stack((map_1_x, map_1_y), axis=-1).astype(np.float32)
        self.map_1 = torch.as_tensor(map_1, dtype=torch.float32, device='cuda')

    def rectify_cpu(self, image_0:np.ndarray, image_1:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
        image_0 = cv2.remap(image_0, self.map_0_x, self.map_0_y, cv2.INTER_LINEAR)
        image_1 = cv2.remap(image_1, self.map_1_x, self.map_1_y, cv2.INTER_LINEAR)

        return image_0, image_1

    def rectify_gpu(self, image_0:torch.Tensor, image_1:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        cv_image_0 = cvcuda.remap(
            src=cvcuda.as_tensor(image_0, "HWC"),
            map=cvcuda.as_tensor(self.map_0, "HWC"),
            src_interp=cvcuda.Interp.LINEAR,
            map_interp=cvcuda.Interp.NEAREST,
            map_type=cvcuda.Remap.ABSOLUTE,
            border=cvcuda.Border.CONSTANT,
        )

        cv_image_1 = cvcuda.remap(
            src=cvcuda.as_tensor(image_1, "HWC"),
            map=cvcuda.as_tensor(self.map_1, "HWC"),
            src_interp=cvcuda.Interp.LINEAR,
            map_interp=cvcuda.Interp.NEAREST,
            map_type=cvcuda.Remap.ABSOLUTE,
            border=cvcuda.Border.CONSTANT,
        )

        image_tensor_0 = torch.as_tensor(cv_image_0.cuda(), device='cuda').to(torch.uint8)
        image_tensor_1 = torch.as_tensor(cv_image_1.cuda(), device='cuda').to(torch.uint8)
        
        return image_tensor_0, image_tensor_1



if __name__ == "__main__":

    from pathlib import Path
    import pickle
    import time

    calibration_data_path = Path("/home/charles/repos/mv_app/tools/cam_calibration/calibration_data.pickle")

    calibration_data_dict = pickle.loads(calibration_data_path.read_bytes())

    map_x_0 = calibration_data_dict["map_x_0"]
    map_y_0 = calibration_data_dict["map_y_0"]
    map_x_1 = calibration_data_dict["map_x_1"]
    map_y_1 = calibration_data_dict["map_y_1"]

    stereo_rectifier = StereoRectifier(map_x_0, map_y_0, map_x_1, map_y_1)

    image_path = "/home/charles/repos/mv_app/tools/cam_calibration/images/1792802281_640x480.jpg"
    image = cv2.imread(image_path)

    # CPU benchmark
    for _ in range(50):
        stereo_rectifier.rectify_cpu(image, image)
        
    t0 = time.perf_counter()
    for _ in range(100):
        rect_image, _ = stereo_rectifier.rectify_cpu(image, image)
    print("CPU: ", (time.perf_counter() - t0) * 1000, "ms")
    cv2.imwrite("rectified-cpu.jpg", rect_image)
    

    # GPU benchmark
    image_tensor = torch.as_tensor(image, dtype=torch.float32, device='cuda')
    with cvcuda.Stream():
        for _ in range(50):
            
            stereo_rectifier.rectify_gpu(image_tensor, image_tensor)
        cvcuda.Stream().sync()
    
        t0 = time.perf_counter()
        for _ in range(100):
            t_image_0, _ = stereo_rectifier.rectify_gpu(image_tensor, image_tensor)
            cvcuda.Stream().sync()
    print("GPU: ", (time.perf_counter() - t0) * 1000, "ms")
    rect_image = t_image_0.cpu().numpy()
    cv2.imwrite("rectified-gpu.jpg", rect_image)


    