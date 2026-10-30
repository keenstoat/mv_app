import cv2
import numpy as np
from pathlib import Path
import pickle

CHECKERBOARD_GRID_SIZE = (9, 6)
CHECKERBOARD_SQUARE_LENGTH_MM = 24.5

def serialize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (np.float32, np.float64)):
        return float(value)
    return value
  
def get_corners(images:list[np.ndarray]):
    # termination criteria
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    images_corners = [] # 2d points in image plane.

    win_size = (5, 5)
    zero_zone = (-1, -1)
    for image in images:

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        success, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_GRID_SIZE, None)

        if success:
            corners = cv2.cornerSubPix(gray, corners, win_size, zero_zone, criteria)
            images_corners.append(corners)
        else:
            images_corners.append(None)

    return images_corners

def get_points(images_sbs_path:Path):

    grid_size = CHECKERBOARD_GRID_SIZE
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    obj_points = np.zeros((grid_size[1] * grid_size[0], 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1,2) * CHECKERBOARD_SQUARE_LENGTH_MM

    # Arrays to store object points and image points from all the images.
    obj_points_list = [] # 3d point in real world space
    image_0_points = [] # 2d points in image plane.
    image_1_points = [] # 2d points in image plane.

    for image_path in images_sbs_path.glob("**/*.jpg"):
        if image_path.parent.stem.endswith("_skip"):
            continue
        image_sbs = cv2.imread(image_path)

        slices = 2
        height = image_sbs.shape[0]
        width = image_sbs.shape[1] // slices
        corners_sbs = get_corners([image_sbs[:, :width], image_sbs[:, width:]])
        if all(corners is not None for corners in corners_sbs):
            obj_points_list.append(obj_points)
            image_0_points.append(corners_sbs[0])
            image_1_points.append(corners_sbs[1])
        else:
            print(f"At least one corners not calculated in {image_path.name}")

    return obj_points_list, image_0_points, image_1_points, height, width

def get_calibration_maps(images_sbs_dir_path:Path):

    
    obj_points_list, image_0_points, image_1_points, height, width = get_points(images_sbs_dir_path)

    image_size = (width, height)

    title = "="*20 + f" Generating maps for WxH = {width}x{height} " + "="*20
    print(title)

    rms_cam_0, mtx_0, dist_0, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_list, image_0_points, image_size, None, None
    )
    new_mtx_0, roi = cv2.getOptimalNewCameraMatrix(mtx_0, dist_0, image_size, 1, image_size)

    print(f"Calibration error - camera 0: {rms_cam_0} px")

    rms_cam_1, mtx_1, dist_1, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_list, image_1_points, image_size, None, None
    )
    new_mtx_1, roi = cv2.getOptimalNewCameraMatrix(mtx_1, dist_1, image_size, 1, image_size)

    print(f"Calibration error - camera 1: {rms_cam_1} px")

    flags = cv2.CALIB_FIX_INTRINSIC

    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    (
        rms_stereo, 
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
        ) # pyright: ignore
    print(f"Calibration error Stereo: {rms_stereo} px")

    # Compute rectification parameters
    # alpha=0 removes black margins but crops; alpha=1 preserves all pixels with some black padding
    alpha = 0 

    R0, R1, P0, P1, Q, validRoi1, validRoi2 = cv2.stereoRectify(
        new_mtx_0, dist_0, 
        new_mtx_1, dist_1, 
        image_size,         # Tuple: (width, height)
        rot, trans,         # Matrices obtained from cv2.stereoCalibrate
        alpha=alpha
    )

    # Initialize maps for Left Camera
    map_0_x, map_0_y = cv2.initUndistortRectifyMap(
        new_mtx_0, dist_0, R0, P0, image_size, cv2.CV_32FC1
    )

    # Initialize maps for Right Camera
    map_1_x, map_1_y = cv2.initUndistortRectifyMap(
        new_mtx_1, dist_1, R1, P1, image_size, cv2.CV_32FC1
    )

    # R0/R1: rectification rotation matrices that transform cam0 and cam1 coordinate frames to the rectified frame.
    # P0/P1: projection matrices mapping 3D points in the rectified frame straight to 2D image coordinates.

    baseline_raw = abs(trans[0][0])
    baseline_rectified = -P1[0, 3] / P1[0, 0]
    focal_x, focal_y = P0[0, 0], P0[1, 1]
    print()
    print(f"baseline rect (mm): {baseline_rectified:.2f}")
    print(f"Baseline raw (mm) : {baseline_raw:.2f}") # distance between camera lenses
    print(f"focal length (px):  {focal_x:.2f} - {focal_y:.2f}")

    # The re-projection error measures how accurately a camera calibration or 3D reconstruction model matches the real world. It measures the distance (in pixels) between where a 3D feature point actually appears in an image versus where the mathematical camera model predicts it should appear.
    data = {
        "rms_cam_0": rms_cam_0, # RMS re-projection error in px
        "rms_cam_1": rms_cam_1, # RMS re-projection error in px
        "rms_stereo": rms_stereo, # RMS re-projection error in px
        "height_px": height,
        "width_px": width,
        "baseline_unit": "mm",
        "baseline_raw": baseline_raw, # distance between camera lenses in mm bc the checkerboard is in mm
        "baseline_rect": baseline_rectified, # same but from the rectified images
        "focal_length_unit": "px",
        "focal_length_x" : focal_x, # in pixels
        "focal_length_y" : focal_y, # in pixels
        "R0": R0, # rotation matrix for camera 0
        "R1": R1, # rotation matrix for camera 1
        "P0": P0, # projection matrix for camera 0
        "P1": P1, # projection matrix for camera 1
        "map_0_x": map_0_x, 
        "map_0_y": map_0_y, 
        "map_1_x": map_1_x, 
        "map_1_y": map_1_y 
    }
    # serializable_data = {k: serialize(v) for k, v in data.items()}
    # json_data = json.dumps(serializable_data)

    byte_data = pickle.dumps(data)
    
    base_dir = Path(__file__).parent.absolute()

    calibration_filepath = base_dir/f"stereo_rectify_{width}x{height}.pickle"
    with open(calibration_filepath, "wb") as calibration_file:
        calibration_file.write(byte_data)

    print("\nCalibration data saved to file: ", calibration_filepath)
    print("=" * len(title))
    print()

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

    _base_dir = Path(__file__).parent.absolute()
    _calibration_images_dir = _base_dir / "_images" / "640x480"
    get_calibration_maps(_calibration_images_dir)

    _calibration_images_dir = _base_dir / "_images" / "1024x576"
    get_calibration_maps(_calibration_images_dir)



    # stereo_estimator = get_stereo_estimator()

    # image_0, image_1 = None, None # put images here
    
    # disparity_map = get_disparity_map(
    #     stereo_estimator, 
    #     image_0, map_x_0, map_y_0,
    #     image_1, map_x_1, map_y_1,
    # )

    # result_image = get_depth_color_map(disparity_map)

