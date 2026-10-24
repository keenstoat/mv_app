import cv2
import numpy as np


class Triangulate:

    def __init__(self, projection_mat_0, projection_mat_1) -> None:

        self.projection_mat_0 = projection_mat_0
        self.projection_mat_1 = projection_mat_1
    
    def triangulate_object(self, point_0, point_1):

        # notice the input points are [x, y]
        # point_0 = np.array([
        #     (box_0[0] + box_0[2]) / 2,
        #     (box_0[1] + box_0[3]) / 2
        # ])
        # but the triangulate requries [[x], [y]]
        # 
        
        # Convert to 2x1 arrays
        pts_0 = np.asarray(point_0, dtype=np.float64).reshape(2, 1)
        pts_1 = np.asarray(point_1, dtype=np.float64).reshape(2, 1)
    
        # Triangulate
        point_4d = cv2.triangulatePoints(
            self.projection_mat_0, self.projection_mat_1, pts_0, pts_1
        )
    
        # Convert homogeneous to Euclidean coordinates
        point_3d = point_4d[:3] / point_4d[3]
    
        return point_3d.flatten()