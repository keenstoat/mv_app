import cv2
import numpy as np
from pathlib import Path

def get_corners(images:list[np.ndarray], grid_size:tuple[int, int]):
    # termination criteria
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    images_corners = [] # 2d points in image plane.

    win_size = (5, 5)
    zero_zone = (-1, -1)
    for image in images:

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        success, corners = cv2.findChessboardCorners(gray, grid_size, None)

        if success:
            corners = cv2.cornerSubPix(gray, corners, win_size, zero_zone, criteria)
            images_corners.append(corners)
        else:
            images_corners.append(None)

    return images_corners

def get_points(images_sbs_path:Path):

    grid_size = (9, 6)

    square_length = 24.5 # mm

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    obj_points = np.zeros((grid_size[1] * grid_size[0], 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1,2) * square_length

    # Arrays to store object points and image points from all the images.
    obj_points_list = [] # 3d point in real world space
    image_0_points = [] # 2d points in image plane.
    image_1_points = [] # 2d points in image plane.

    for image_path in images_sbs_path.iterdir():

        image_sbs = cv2.imread(image_path)

        slices = 2
        height = image_sbs.shape[0]
        width = image_sbs.shape[1] // slices
        corners_sbs = get_corners([image_sbs[:, :width], image_sbs[:, width:]], grid_size)
        if all([corners is not None for corners in corners_sbs]):
            obj_points_list.append(obj_points)
            image_0_points.append(corners_sbs[0])
            image_1_points.append(corners_sbs[1])
        else:
            print(f"At least one corners not calculated in {image_path.name}")

    return obj_points_list, image_0_points, image_1_points, height, width

def get_calibration_maps(images_sbs_path:Path):


    obj_points_list, image_0_points, image_1_points, height, width = get_points(images_sbs_path)

    image_size = (width, height)

    ret, mtx_0, dist_0, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_list, image_0_points, image_size, None, None
    )
    new_mtx_0, roi = cv2.getOptimalNewCameraMatrix(mtx_0, dist_0, image_size, 1, image_size)

    print("ERROR 0: ", ret)

    ret, mtx_1, dist_1, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_list, image_1_points, image_size, None, None
    )
    new_mtx_1, roi = cv2.getOptimalNewCameraMatrix(mtx_1, dist_1, image_size, 1, image_size)

    print("ERROR 1: ", ret)


    header = ['Rotation','Translation', 'ProjectionLeft', 'ProjectionRight'] # for the csv file

    flags = cv2.CALIB_FIX_INTRINSIC

    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    (
        retStereo, 
        new_mtx_0, 
        dist_0, 
        new_mtx_1, 
        dist_1, 
        rot, 
        trans, 
        essentialMatrix, 
        fundamentalMatrix
    ) = cv2.stereoCalibrate(
            obj_points_list, 
            image_0_points, image_1_points, 
            new_mtx_0, dist_0, 
            new_mtx_1, dist_1, 
            image_size, 
            criteria_stereo, 
            flags
        )
    print("ERROR Stereo: ", retStereo)

    print(f"Baseline: {abs(trans[0][0]):.2f} mm") # distance between camera lenses

    # Compute rectification parameters
    # alpha=0 removes black margins but crops; alpha=1 preserves all pixels with some black padding
    alpha = 0 

    R0, R1, P0, P1, Q, validRoi1, validRoi2 = cv2.stereoRectify(
        new_mtx_0, dist_0, 
        new_mtx_1, dist_1, 
        image_size,   # Tuple: (width, height)
        rot, trans,         # Matrices obtained from cv2.stereoCalibrate
        alpha=alpha
    )

    # Initialize maps for Left Camera
    map_x_0, map_y_0 = cv2.initUndistortRectifyMap(
        new_mtx_0, dist_0, R0, P0, image_size, cv2.CV_32FC1
    )

    # Initialize maps for Right Camera
    map_x_1, map_y_1 = cv2.initUndistortRectifyMap(
        new_mtx_1, dist_1, R1, P1, image_size, cv2.CV_32FC1
    )

    return map_x_0, map_y_0, map_x_1, map_y_1

def get_stereo_estimator() -> cv2.StereoSGBM:
    min_disp = 0
    num_disp = 64 - min_disp  # Max search range for depth. Increase for closer objects.
    block_size = 5            # Matched block size. Must be an odd number (3, 5, 7, etc.)

    stereo = cv2.StereoSGBM.create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        # P1 and P2 control depth smoothness. These formulas are standard baselines:
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,      # Maximum allowed difference in left-right consistency check
        uniquenessRatio=15,   # Margin in % by which the best match must beat the second best
        speckleWindowSize=100, # Reject small speckles/noise patches smaller than this size
        speckleRange=2,       # Maximum disparity variation within a noise patch
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY # Faster execution on edge devices like Jetson
    )
    return stereo

def get_disparity_map(stereo_estimator:cv2.StereoSGBM, image_0, map_x_0, map_y_0, image_1, map_x_1, map_y_1):

    image_0 = cv2.remap(image_0, map_x_0, map_y_0, cv2.INTER_LINEAR)
    image_1 = cv2.remap(image_1, map_x_1, map_y_1, cv2.INTER_LINEAR)

    image_0 = cv2.cvtColor(image_0, cv2.COLOR_RGB2GRAY)
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)

    # compute the raw disparity map
    # The output is a 16-bit signed integer map multiplied internally by 16
    disparity_raw:np.ndarray = stereo_estimator.compute(image_0, image_1)

    # convert to 32-bit float map and undo the internal x16 scaling
    disparity = disparity_raw.astype(np.float32) / 16.0

    # normalize the map to  0-255 range so it can be viewed as an image
    # Filter out invalid negative disparity pixels (areas the camera couldn't see around corners)
    disparity[disparity < 0] = 0

    # Scale to 0-255 integer format
    cv2.normalize(disparity, disparity, 0, 255, cv2.NORM_MINMAX)

    return disparity.astype(np.uint8)

def get_depth_color_map(disparity_map:np.ndarray):
    
    # Jet map scales: Closer objects will appear bright red/orange, far objects blue
    depth_colormap = cv2.applyColorMap(disparity_map, cv2.COLORMAP_JET)
    return cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)

if __name__ == "__main__":

    # calibrate(Path("/home/charles/repos/mv_app/_calibration/1280x720"))

    get_calibration_maps(Path("/home/charles/repos/mv_app/_calibration/640x480"))