import numpy as np
import cv2
import torch
import cvcuda
import vpi
import pickle
from pathlib import Path
from .frame import Frame

class StereoDepth:

    def __init__(self, calibration_data_filepath:str) -> None:

        # the rectification maps were calculated using opencv initUndistortRectifyMap() 
        # using np arrays and so have a HWC layout. 
        # image tensors must have the same layout for the remap to work!
        calibration_data_path = Path(calibration_data_filepath)
        calibration_data_dict = pickle.loads(calibration_data_path.read_bytes())
    
        map_0_x = calibration_data_dict["map_0_x"]
        map_0_y = calibration_data_dict["map_0_y"]
        map_1_x = calibration_data_dict["map_1_x"]
        map_1_y = calibration_data_dict["map_1_y"]

        # used for GPU rectification
        map_0 = np.stack((map_0_x, map_0_y), axis=-1).astype(np.float32)
        self.map_0 = torch.as_tensor(map_0, dtype=torch.float32, device='cuda')

        map_1 = np.stack((map_1_x, map_1_y), axis=-1).astype(np.float32)
        self.map_1 = torch.as_tensor(map_1, dtype=torch.float32, device='cuda')

        self.map_list = [self.map_0, self.map_1]

        # used for triangulation
        self.projection_mat_0 = calibration_data_dict["P0"]
        self.projection_mat_1 = calibration_data_dict["P1"]

    def rectify_gpu(self, frame:Frame):
        """
        Calculates the stereo rectification of the image tensors.
        The rectification is done over the torch tensors on gpu.
        """

        with frame.time_it("rectify"):
            
            for idx, map in enumerate(self.map_list):
                image_tensor = cvcuda.remap(
                    src=cvcuda.as_tensor(frame.image_tensors[idx], frame.layout),
                    map=cvcuda.as_tensor(map, frame.layout),
                    src_interp=cvcuda.Interp.LINEAR,
                    map_interp=cvcuda.Interp.NEAREST,
                    map_type=cvcuda.Remap.ABSOLUTE,
                    border=cvcuda.Border.CONSTANT,
                )
                # replace the image tensors because this is now the image source of truth
                image_tensor = torch.as_tensor(image_tensor.cuda(), dtype=torch.uint8, device='cuda')
                frame.image_tensors[idx] = image_tensor
                frame.is_rectified = True

    def triangulate(self, frame:Frame):

        if frame.detections_batch and frame.detections_batch.detections_list[0].tracker_id.size == 0: 
            return 
            
        with frame.time_it("triangulate"):
            # input points are bounding boxes are shaped N,4
            # [[x,y,x,y], 
            #  [x,y,x,y],
            #  [x,y,x,y]]
            xyxy_0 = frame.detections_batch.detections_list[0].xyxy # pyright: ignore
            xyxy_1 = frame.detections_batch.detections_list[1].xyxy # pyright: ignore
            # now get the center ponits to triangulate - shaped N,2
            # [[x,y], 
            #  [x,y],
            #  [x,y]]
            cxy_0 = (xyxy_0[:, :2] + xyxy_0[:, 2:]) / 2
            cxy_1 = (xyxy_1[:, :2] + xyxy_1[:, 2:]) / 2
            
            # transpose to fit triangulate - shaped 2, N
            # [[x,x,x], 
            # [y,y,y]]
            cxy_0 = cxy_0.T
            cxy_1 = cxy_1.T
            
            # Triangulate - this returns the shape 4, N
            # [[x,x,x], 
            #  [y,y,y],
            #  [z,z,z],
            #  [w,w,w]]
            xyzw = cv2.triangulatePoints(
                self.projection_mat_0, self.projection_mat_1, 
                cxy_0, cxy_1
            )  # pyright: ignore
        
            # Convert homogeneous to Euclidean coordinates
            xyz = xyzw[:3, :] / xyzw[3, :]
    
            # return them in shape N, 3
            # [[x,y,z], 
            #  [x,y,z],
            #  [x,y,z]]
            frame.detections_batch.detections_list[0].xyz = xyz.T
            frame.detections_batch.detections_list[1].xyz = xyz.T

    def dense_disparity(self, frame:Frame, apply_jet_colormap=False):
        """
        Calculates the disparity map for the tensor images and creates a grayscale depth map image
        Uses VPI with CUDA backend
        Ref: 
            - https://docs.nvidia.com/vpi/sample_stereo.html
            - https://docs.nvidia.com/vpi/python/build/vpi.stereodisp.html
        """
        def jet_colormap(disparity_u8_tensor):
            """
            Applies the JET colormap to a uint8 [H, W] PyTorch tensor entirely on the GPU.
            Returns a [H, W, 3] uint8 BGR tensor compatible with OpenCV display.
            """
            # Normalize 0-255 to 0.0-1.0 float for mapping math
            x = disparity_u8_tensor.float() / 255.0
            
            # Calculate JET color curves mathematically
            # Blue curve
            b = torch.clamp(1.5 - torch.abs(4.0 * x - 1.0), 0.0, 1.0)
            # Green curve
            g = torch.clamp(1.5 - torch.abs(4.0 * x - 2.0), 0.0, 1.0)
            # Red curve
            r = torch.clamp(1.5 - torch.abs(4.0 * x - 3.0), 0.0, 1.0)
            
            # Stack them into a BGR tensor (matching OpenCV's channel expectation)
            # and convert back to uint8 (0-255)
            rgb_tensor = (torch.stack([r, g, b], dim=-1) * 255.0).to(torch.uint8)
            
            return rgb_tensor
            
        with frame.time_it("disparity"):
            
            vpi_tensor_0 = vpi.asimage(frame.image_tensors[0])
            vpi_tensor_1 = vpi.asimage(frame.image_tensors[1])

            vpi_tensor_0 = vpi_tensor_0.convert(vpi.Format.U8, scale=1, backend=vpi.Backend.CUDA)
            vpi_tensor_1 = vpi_tensor_1.convert(vpi.Format.U8, scale=1, backend=vpi.Backend.CUDA)
    
            MAX_DISPARITY = 64

            # disparity map values are S16 = signed 16-bit integers
            vpi_disparity_map = vpi.stereodisp(
                vpi_tensor_0, 
                vpi_tensor_1, 
                window=5,
                includediagonals=True,
                quality=8, # 1-8
                maxdisp=MAX_DISPARITY,
                backend=vpi.Backend.CUDA
            )

            # downscale to U8 = unsigned 8-bit, same as uint8
            vpi_disparity_map = vpi_disparity_map.convert(
                vpi.Format.U8, 
                scale=1.0/(32*MAX_DISPARITY)*255,
                backend=vpi.Backend.CUDA
            )
            
            # zero-copy to a torch tensor
            vpi_disparity_map = torch.as_tensor(
                vpi_disparity_map.cuda(), dtype=torch.uint8, device='cuda'
            )

            frame.dense_disparity_depth = jet_colormap(vpi_disparity_map) if apply_jet_colormap else vpi_disparity_map
        